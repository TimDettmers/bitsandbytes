#!/usr/bin/env python

import argparse
import re

from_line_re = re.compile(r"FROM\s+(?P<image>\S+)\s+AS\s+(?P<target>\S+)")


def find_image_in_dockerfile(dockerfile, target):
    with open(dockerfile) as f:
        for line in f.readlines():
            if (m := from_line_re.match(line)) and m.group("target") == target:
                return m.group("image")
    raise ValueError(f"Target {target} not defined in {dockerfile}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dockerfile")
    ap.add_argument("target")
    ap.add_argument("-t", "--tag-only", action="store_true")
    ap.add_argument("-v", "--version-only", action="store_true")
    args = ap.parse_args()
    result = find_image_in_dockerfile(args.dockerfile, args.target)
    if args.tag_only or args.version_only:
        result = result.rpartition(":")[-1]
        if args.version_only:
            result = result.split("-")[0]
    print(result)

if __name__ == '__main__':
    main()
