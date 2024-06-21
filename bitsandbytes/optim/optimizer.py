# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import abc as container_abcs, defaultdict
from copy import deepcopy
from itertools import chain

import torch

import bitsandbytes.functional as F


class MockArgs:
    def __init__(self, initial_data):
        for key in initial_data:
            setattr(self, key, initial_data[key])


class GlobalOptimManager:
    """
    A global optimizer manager for enabling custom optimizer configs.
    """

    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        self.pid2config = {}
        self.index2config = {}
        self.optimizer = None
        self.uses_config_override = False
        self.module_weight_config_triple = []

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def register_parameters(self, params):
        param_groups = list(params)
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]

        for group_index, group in enumerate(param_groups):
            for p_index, p in enumerate(group["params"]):
                if id(p) in self.pid2config:
                    self.index2config[(group_index, p_index)] = self.pid2config[id(p)]

    def override_config(self, parameters, key=None, value=None, key_value_dict=None):
        """
        Override initial optimizer config with specific hyperparameters.

        The key-values of the optimizer config for the input parameters are overridden
        This can be both, optimizer parameters like `betas` or `lr`, or it can be
        8-bit specific parameters like `optim_bits` or `percentile_clipping`.

        Arguments:
           parameters (`torch.Tensor` or `list(torch.Tensors)`):
             The input parameters.
           key (`str`):
             The hyperparamter to override.
           value:
             The hyperparameter values.
           key_value_dict (`dict`):
             A dictionary with multiple key-values to override.

        Example:

        ```py
        import torch
        import bitsandbytes as bnb

        mng = bnb.optim.GlobalOptimManager.get_instance()

        model = MyModel()
        mng.register_parameters(model.parameters()) # 1. register parameters while still on CPU

        model = model.cuda()
        # use 8-bit optimizer states for all parameters
        adam = bnb.optim.Adam(model.parameters(), lr=0.001, optim_bits=8)

        # 2. override: the parameter model.fc1.weight now uses 32-bit Adam
        mng.override_config(model.fc1.weight, 'optim_bits', 32)
        ```
        """
        self.uses_config_override = True
        if isinstance(parameters, torch.nn.Parameter):
            parameters = [parameters]
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        if key is not None and value is not None:
            assert key_value_dict is None
            key_value_dict = {key: value}

        if key_value_dict is not None:
            for p in parameters:
                if id(p) in self.pid2config:
                    self.pid2config[id(p)].update(key_value_dict)
                else:
                    self.pid2config[id(p)] = key_value_dict

    def register_module_override(self, module, param_name, config):
        self.module_weight_config_triple.append((module, param_name, config))


class Optimizer8bit(torch.optim.Optimizer):
    def __init__(self, params, defaults, optim_bits=32, is_paged=False):
        """
        Base 8-bit optimizer class.

        Arguments:
            params (`torch.tensor`):
                The input parameters to optimize.
            optim_bits (`int`, defaults to 32):
                The number of bits of the optimizer state.
            is_paged (`bool`, defaults to `False`):
                Whether the optimizer is a paged optimizer or not.
        """
        super().__init__(params, defaults)
        self.initialized = False
        self.name2qmap = {}
        self.is_paged = is_paged
        self.page_mng = F.GlobalPageManager.get_instance()

        self.mng = GlobalOptimManager.get_instance()
        self.non_castable_tensor_keys = {
            "qmap1",
            "qmap2",
            "max1",
            "max2",
            "new_max1",
            "new_max2",
            "state1",
            "state2",
            "gnorm_vec",
            "absmax1",
            "absmax2",
            "unorm_vec",
        }

        if optim_bits == 8:
            self.fill_qmap()

    def fill_qmap(self):
        self.name2qmap["dynamic"] = F.create_dynamic_map(signed=True)
        self.name2qmap["udynamic"] = F.create_dynamic_map(signed=False)

    def __setstate__(self, state):
        super().__setstate__(state)

    def load_state_dict(self, state_dict):
        """Load an optimizer state.

        Arguments:
            state_dict (`dict`):
                An optimizer state (should be returned from a call to `state_dict`) to load.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict["param_groups"]

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of parameter groups")
        param_lens = (len(g["params"]) for g in groups)
        saved_lens = (len(g["params"]) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError(
                "loaded state dict contains a parameter group that doesn't match the size of optimizer's group",
            )

        # Update the state
        id_map = {
            old_id: p
            for old_id, p in zip(
                chain.from_iterable(g["params"] for g in saved_groups),
                chain.from_iterable(g["params"] for g in groups),
            )
        }

        def cast(param, value):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                if param.is_floating_point() and value.dtype != torch.uint8:
                    value = value.to(param.dtype)
                return value
            elif isinstance(value, dict):
                for k, v in value.items():
                    if k in self.non_castable_tensor_keys:
                        value[k] = v.to(param.device)
                    else:
                        value[k] = cast(param, v)

                return value
            elif isinstance(value, container_abcs.Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group["params"] = group["params"]
            return new_group

        param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({"state": state, "param_groups": param_groups})

    def to_gpu(self):
        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group["params"]):
                if p in self.state:
                    values = self.state[p]
                    for k, v in values.items():
                        if isinstance(v, torch.Tensor):
                            is_paged = getattr(v, "is_paged", False)
                            if not is_paged:
                                self.state[p][k] = v.to(p.device)

    def check_overrides(self):
        for module, attr, config in self.mng.module_weight_config_triple:
            pmodule = getattr(module, attr)
            assert pmodule is not None
            assert isinstance(pmodule, torch.Tensor) or isinstance(pmodule, torch.Parameter)
            found = False
            for gindex, group in enumerate(self.param_groups):
                if found:
                    break
                for pindex, p in enumerate(group["params"]):
                    if found:
                        break
                    if id(p) == id(pmodule):
                        # found the matching parameter
                        # init override
                        self.mng.pid2config[id(p)] = config
                        self.mng.index2config[(gindex, pindex)] = self.mng.pid2config[id(p)]
                        found = True

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Arguments:
            closure (`Callable`, *optional*, defaults to `None`):
                A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        overflows = []

        if not self.initialized:
            self.check_overrides()
            self.to_gpu()  # needed for fairseq pure fp16 training
            self.initialized = True

        # if self.is_paged: self.page_mng.prefetch_all()
        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    self.init_state(group, p, gindex, pindex)

                self.prefetch_state(p)
                self.update_step(group, p, gindex, pindex)
                torch.cuda.synchronize()
        if self.is_paged:
            # all paged operation are asynchronous, we need
            # to sync to make sure all tensors are in the right state
            torch.cuda.synchronize()

        return loss

    def get_config(self, gindex, pindex, group):
        config = {}
        config["betas"] = group["betas"]
        config["eps"] = group["eps"]
        config["weight_decay"] = group["weight_decay"]
        config["lr"] = group["lr"]
        config["optim_bits"] = self.args.optim_bits
        config["min_8bit_size"] = self.args.min_8bit_size
        config["percentile_clipping"] = self.args.percentile_clipping
        config["block_wise"] = self.args.block_wise
        config["max_unorm"] = self.args.max_unorm
        config["skip_zeros"] = self.args.skip_zeros

        if (gindex, pindex) in self.mng.index2config:
            config.update(self.mng.index2config[(gindex, pindex)])
        return config

    def init_state(self, group, p, gindex, pindex):
        raise NotImplementedError("init_state method needs to be overridden")

    def update_step(self, group, p, gindex, pindex):
        raise NotImplementedError("The update_step method needs to be overridden")

    def get_state_buffer(self, p, dtype=torch.float32):
        if not self.is_paged or p.numel() < 1e5:
            return torch.zeros_like(p, dtype=dtype, device=p.device)
        else:
            # > 1 MB
            buff = F.get_paged(*p.shape, dtype=dtype, device=p.device)
            F.fill(buff, 0)
            self.page_mng.paged_tensors.append(buff)
            return buff

    def prefetch_state(self, p):
        if self.is_paged:
            state = self.state[p]
            s1 = state["state1"]
            is_paged = getattr(s1, "is_paged", False)
            if is_paged:
                F.prefetch_tensor(state["state1"])
                if "state2" in state:
                    F.prefetch_tensor(state["state2"])


class Optimizer2State(Optimizer8bit):
    def __init__(
        self,
        optimizer_name,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=True,
        max_unorm=0.0,
        skip_zeros=False,
        is_paged=False,
    ):
        """
        Base 2-state update optimizer class.

        Arguments:
            optimizer_name (`str`):
                The name of the optimizer.
            params (`torch.tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-3):
                The learning rate.
            betas (`tuple`, defaults to (0.9, 0.999)):
                The beta values for the optimizer.
            eps (`float`, defaults to 1e-8):
                The epsilon value for the optimizer.
            weight_decay (`float`, defaults to 0.0):
                The weight decay value for the optimizer.
            optim_bits (`int`, defaults to 32):
                The number of bits of the optimizer state.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
            percentile_clipping (`int`, defaults to 100):
                Adapts clipping threshold automatically by tracking the last 100 gradient norms and clipping the gradient at a certain percentile to improve stability.
            block_wise (`bool`, defaults to `True`):
                Whether to independently quantize each block of tensors to reduce outlier effects and improve stability.
            max_unorm (`float`, defaults to 0.0):
                The maximum value to normalize each block with.
            skip_zeros (`bool`, defaults to `False`):
                Whether to skip zero values for sparse gradients and models to ensure correct updates.
            is_paged (`bool`, defaults to `False`):
                Whether the optimizer is a paged optimizer or not.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if isinstance(betas, str):
            # format: '(beta1, beta2)'
            betas = betas.replace("(", "").replace(")", "").strip().split(",")
            betas = [float(b) for b in betas]
        for i in range(len(betas)):
            if not 0.0 <= betas[i] < 1.0:
                raise ValueError(f"Invalid beta parameter at index {i}: {betas[i]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults, optim_bits, is_paged)

        if args is None:
            args = {}
            args["optim_bits"] = optim_bits
            args["percentile_clipping"] = 100
            args["min_8bit_size"] = min_8bit_size
            args["percentile_clipping"] = percentile_clipping
            args["block_wise"] = block_wise
            args["max_unorm"] = max_unorm
            args["skip_zeros"] = skip_zeros

            self.args = MockArgs(args)
        else:
            self.args = args

        self.optimizer_name = optimizer_name

    @torch.no_grad()
    def init_state(self, group, p, gindex, pindex):
        config = self.get_config(gindex, pindex, group)

        if config["optim_bits"] == 32:
            dtype = torch.float32
        elif config["optim_bits"] == 8:
            dtype = torch.uint8
        else:
            raise NotImplementedError(f'Amount of optimizer bits not supported: {config["optim_bits"]}')

        if p.numel() < config["min_8bit_size"]:
            dtype = torch.float32

        state = self.state[p]
        state["step"] = 0

        if dtype == torch.float32:
            state["state1"] = self.get_state_buffer(p, dtype=torch.float32)
            state["state2"] = self.get_state_buffer(p, dtype=torch.float32)
        elif dtype == torch.uint8:
            if state["step"] == 0:
                if "dynamic" not in self.name2qmap:
                    self.fill_qmap()
                self.name2qmap["dynamic"] = self.name2qmap["dynamic"].to(p.device)
                self.name2qmap["udynamic"] = self.name2qmap["udynamic"].to(p.device)

            state["state1"] = self.get_state_buffer(p, dtype=torch.uint8)
            state["qmap1"] = self.name2qmap["dynamic"]

            state["state2"] = self.get_state_buffer(p, dtype=torch.uint8)
            state["qmap2"] = self.name2qmap["udynamic"]

            if config["block_wise"]:
                n = p.numel()
                blocks = n // 2048
                blocks += 1 if n % 2048 > 0 else 0

                state["absmax1"] = torch.zeros((blocks,), dtype=torch.float32, device=p.device)
                state["absmax2"] = torch.zeros((blocks,), dtype=torch.float32, device=p.device)
            else:
                state["max1"] = torch.zeros((1,), dtype=torch.float32, device=p.device)
                state["new_max1"] = torch.zeros((1,), dtype=torch.float32, device=p.device)
                state["max2"] = torch.zeros((1,), dtype=torch.float32, device=p.device)
                state["new_max2"] = torch.zeros((1,), dtype=torch.float32, device=p.device)

        if config["percentile_clipping"] < 100:
            state["gnorm_vec"] = torch.zeros((100,), device=p.device)

        if config["max_unorm"] > 0.0:
            state["unorm_vec"] = torch.zeros((1,), device=p.device)

    @torch.no_grad()
    def update_step(self, group, p, gindex, pindex):
        state = self.state[p]
        grad = p.grad

        config = self.get_config(gindex, pindex, group)

        state["step"] += 1
        step = state["step"]

        if config["percentile_clipping"] < 100:
            current_gnorm, clip_value, gnorm_scale = F.percentile_clipping(
                grad,
                state["gnorm_vec"],
                step,
                config["percentile_clipping"],
            )
        else:
            gnorm_scale = 1.0

        if state["state1"].dtype == torch.float:
            F.optimizer_update_32bit(
                self.optimizer_name,
                grad,
                p,
                state["state1"],
                config["betas"][0],
                config["eps"],
                step,
                config["lr"],
                state["state2"],
                config["betas"][1],
                config["weight_decay"],
                gnorm_scale,
                state["unorm_vec"] if config["max_unorm"] > 0.0 else None,
                max_unorm=config["max_unorm"],
                skip_zeros=config["skip_zeros"],
            )

        elif state["state1"].dtype == torch.uint8 and not config["block_wise"]:
            F.optimizer_update_8bit(
                self.optimizer_name,
                grad,
                p,
                state["state1"],
                state["state2"],
                config["betas"][0],
                config["betas"][1],
                config["eps"],
                step,
                config["lr"],
                state["qmap1"],
                state["qmap2"],
                state["max1"],
                state["max2"],
                state["new_max1"],
                state["new_max2"],
                config["weight_decay"],
                gnorm_scale=gnorm_scale,
                unorm_vec=state["unorm_vec"] if config["max_unorm"] > 0.0 else None,
                max_unorm=config["max_unorm"],
            )

            # swap maxes
            state["max1"], state["new_max1"] = state["new_max1"], state["max1"]
            state["max2"], state["new_max2"] = state["new_max2"], state["max2"]
        elif state["state1"].dtype == torch.uint8 and config["block_wise"]:
            F.optimizer_update_8bit_blockwise(
                self.optimizer_name,
                grad,
                p,
                state["state1"],
                state["state2"],
                config["betas"][0],
                config["betas"][1],
                config["eps"],
                step,
                config["lr"],
                state["qmap1"],
                state["qmap2"],
                state["absmax1"],
                state["absmax2"],
                config["weight_decay"],
                gnorm_scale=gnorm_scale,
                skip_zeros=config["skip_zeros"],
            )


class Optimizer1State(Optimizer8bit):
    def __init__(
        self,
        optimizer_name,
        params,
        lr=1e-3,
        betas=(0.9, 0.0),
        eps=1e-8,
        weight_decay=0.0,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=True,
        max_unorm=0.0,
        skip_zeros=False,
        is_paged=False,
    ):
        """
        Base 1-state update optimizer class.

        Arguments:
            optimizer_name (`str`):
                The name of the optimizer.
            params (`torch.tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-3):
                The learning rate.
            betas (`tuple`, defaults to (0.9, 0.0)):
                The beta values for the optimizer.
            eps (`float`, defaults to 1e-8):
                The epsilon value for the optimizer.
            weight_decay (`float`, defaults to 0.0):
                The weight decay value for the optimizer.
            optim_bits (`int`, defaults to 32):
                The number of bits of the optimizer state.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
            percentile_clipping (`int`, defaults to 100):
                Adapts clipping threshold automatically by tracking the last 100 gradient norms and clipping the gradient at a certain percentile to improve stability.
            block_wise (`bool`, defaults to `True`):
                Whether to independently quantize each block of tensors to reduce outlier effects and improve stability.
            max_unorm (`float`, defaults to 0.0):
                The maximum value to normalize each block with.
            skip_zeros (`bool`, defaults to `False`):
                Whether to skip zero values for sparse gradients and models to ensure correct updates.
            is_paged (`bool`, defaults to `False`):
                Whether the optimizer is a paged optimizer or not.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        for i in range(len(betas)):
            if not 0.0 <= betas[i] < 1.0:
                raise ValueError(f"Invalid beta parameter at index {i}: {betas[i]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults, optim_bits, is_paged)

        if args is None:
            args = {}
            args["optim_bits"] = optim_bits
            args["percentile_clipping"] = 100
            args["min_8bit_size"] = min_8bit_size
            args["percentile_clipping"] = percentile_clipping
            args["block_wise"] = block_wise
            args["max_unorm"] = max_unorm
            args["skip_zeros"] = skip_zeros

            self.args = MockArgs(args)
        else:
            self.args = args

        self.optimizer_name = optimizer_name

    @torch.no_grad()
    def init_state(self, group, p, gindex, pindex):
        config = self.get_config(gindex, pindex, group)

        if config["optim_bits"] == 32:
            dtype = torch.float32
        elif config["optim_bits"] == 8:
            dtype = torch.uint8
        else:
            raise NotImplementedError(f'Amount of optimizer bits not supported: {config["optim_bits"]}')

        if p.numel() < config["min_8bit_size"]:
            dtype = torch.float32

        state = self.state[p]
        state["step"] = 0

        if dtype == torch.float32 or (dtype == torch.uint8 and p.numel() < 4096):
            state["state1"] = self.get_state_buffer(p, dtype=torch.float32)
        elif dtype == torch.uint8:
            if state["step"] == 0:
                if "dynamic" not in self.name2qmap:
                    self.fill_qmap()
                self.name2qmap["dynamic"] = self.name2qmap["dynamic"].to(p.device)

            state["state1"] = self.get_state_buffer(p, dtype=torch.uint8)
            state["qmap1"] = self.name2qmap["dynamic"]

            if config["block_wise"]:
                n = p.numel()
                blocks = n // 2048
                blocks += 1 if n % 2048 > 0 else 0

                state["absmax1"] = torch.zeros((blocks,), dtype=torch.float32, device=p.device)
            else:
                state["max1"] = torch.zeros((1,), dtype=torch.float32, device=p.device)
                state["new_max1"] = torch.zeros((1,), dtype=torch.float32, device=p.device)

        if config["percentile_clipping"] < 100:
            state["gnorm_vec"] = torch.zeros((100,), device=p.device)

        if config["max_unorm"] > 0.0:
            state["unorm_vec"] = torch.zeros((1,), device=p.device)

    @torch.no_grad()
    def update_step(self, group, p, gindex, pindex):
        state = self.state[p]
        grad = p.grad

        config = self.get_config(gindex, pindex, group)

        state["step"] += 1
        step = state["step"]

        if config["percentile_clipping"] < 100:
            current_gnorm, clip_value, gnorm_scale = F.percentile_clipping(
                grad,
                state["gnorm_vec"],
                step,
                config["percentile_clipping"],
            )
        else:
            gnorm_scale = 1.0

        if state["state1"].dtype == torch.float:
            F.optimizer_update_32bit(
                self.optimizer_name,
                grad,
                p,
                state["state1"],
                config["betas"][0],
                config["eps"],
                step,
                config["lr"],
                None,
                config["betas"][1],
                config["weight_decay"],
                gnorm_scale,
                state["unorm_vec"] if config["max_unorm"] > 0.0 else None,
                max_unorm=config["max_unorm"],
                skip_zeros=config["skip_zeros"],
            )

        elif state["state1"].dtype == torch.uint8 and not config["block_wise"]:
            F.optimizer_update_8bit(
                self.optimizer_name,
                grad,
                p,
                state["state1"],
                None,
                config["betas"][0],
                config["betas"][1],
                config["eps"],
                step,
                config["lr"],
                state["qmap1"],
                None,
                state["max1"],
                None,
                state["new_max1"],
                None,
                config["weight_decay"],
                gnorm_scale,
                state["unorm_vec"] if config["max_unorm"] > 0.0 else None,
                max_unorm=config["max_unorm"],
            )

            state["max1"], state["new_max1"] = state["new_max1"], state["max1"]
        elif state["state1"].dtype == torch.uint8 and config["block_wise"]:
            F.optimizer_update_8bit_blockwise(
                self.optimizer_name,
                grad,
                p,
                state["state1"],
                None,
                config["betas"][0],
                config["betas"][1],
                config["eps"],
                step,
                config["lr"],
                state["qmap1"],
                None,
                state["absmax1"],
                None,
                config["weight_decay"],
                gnorm_scale=gnorm_scale,
                skip_zeros=config["skip_zeros"],
            )
