#pragma once

#include "Algo-Direct-Common.h"

namespace BinSearch {
namespace Details {

template <typename T, Algos A>
struct AlgoScalarBase<T, A, typename std::enable_if<DirectAux::IsDirect2<A>::value>::type> : DirectAux::DirectInfo<2, T, A>
{
private:
    typedef DirectAux::DirectInfo<2, T, A> base_t;
    static const size_t Offset=2;

public:
    AlgoScalarBase(const T* x, const uint32 n)
        : base_t(x, n)
    {
    }

    FORCE_INLINE uint32 scalar(T z) const
    {
        const T* px = base_t::data.xi;
        const uint32* buckets = reinterpret_cast<const uint32 *>(base_t::data.buckets);
        uint32 bidx = base_t::fun_t::f(base_t::data.scaler, base_t::data.cst0, z);
        uint32 iidx = buckets[bidx];
        px += iidx;
        if (z < *px)
            --iidx;
        if (z < *(px+1))
            --iidx;
        return iidx;
    }
};


template <InstrSet I, typename T, Algos A>
struct AlgoVecBase<I, T, A, typename std::enable_if<DirectAux::IsDirect2<A>::value>::type> : AlgoScalarBase<T, A>
{
    static const uint32 nElem = sizeof(typename InstrFloatTraits<I, T>::vec_t) / sizeof(T);

    typedef FVec<I, T> fVec;
    typedef IVec<SSE, T> i128;

    struct Constants
    {
        fVec vscaler;
        fVec vcst0;
        IVec<I, T> one;
    };

private:
    typedef AlgoScalarBase<T, A> base_t;

    FORCE_INLINE
        //NO_INLINE
        void resolve(const FVec<SSE, float>& vz, const IVec<SSE, float>& bidx, uint32 *pr) const
    {
        union U {
            __m128i vec;
            uint32 ui32[4];
        } u;

        const uint32* buckets = reinterpret_cast<const uint32 *>(base_t::data.buckets);
        const float *xi = base_t::data.xi;

        // read indices t
        const double *p3 = reinterpret_cast<const double *>(&xi[(u.ui32[3] = buckets[bidx.get3()])]);
        const double *p2 = reinterpret_cast<const double *>(&xi[(u.ui32[2] = buckets[bidx.get2()])]);
        const double *p1 = reinterpret_cast<const double *>(&xi[(u.ui32[1] = buckets[bidx.get1()])]);
        const double *p0 = reinterpret_cast<const double *>(&xi[(u.ui32[0] = buckets[bidx.get0()])]);

#if 0
        // read pairs ( X(t-1), X(t) )
        __m128 xp3 = _mm_castpd_ps(_mm_load_sd(p3));
        __m128 xp2 = _mm_castpd_ps(_mm_load_sd(p2));
        __m128 xp1 = _mm_castpd_ps(_mm_load_sd(p1));
        __m128 xp0 = _mm_castpd_ps(_mm_load_sd(p0));

        // build:
        // { X(t(0)-1), X(t(1)-1), X(t(2)-1), X(t(3)-1) }
        // { X(t(0)),   X(t(1)),   X(t(2)),   X(t(3)) }
        __m128 h13 = _mm_shuffle_ps(xp1, xp3, (1 << 2) + (1 << 6));
        __m128 h02 = _mm_shuffle_ps(xp0, xp2, (1 << 2) + (1 << 6));
        __m128 u01 = _mm_unpacklo_ps(h02, h13);
        __m128 u23 = _mm_unpackhi_ps(h02, h13);
        __m128 vxm = _mm_shuffle_ps(u01, u23, (0) + (1 << 2) + (0 << 4) + (1 << 6));
        __m128 vxp = _mm_shuffle_ps(u01, u23, (2) + (3 << 2) + (2 << 4) + (3 << 6));
#else
        __m128 xp23 = _mm_castpd_ps(_mm_set_pd(*p3, *p2));
        __m128 xp01 = _mm_castpd_ps(_mm_set_pd(*p1, *p0));
        __m128 vxm = _mm_shuffle_ps(xp01, xp23, (0) + (2 << 2) + (0 << 4) + (2 << 6));
        __m128 vxp = _mm_shuffle_ps(xp01, xp23, (1) + (3 << 2) + (1 << 4) + (3 << 6));
#endif
        IVec<SSE, float> i(u.vec);
        IVec<SSE, float> vlem = operator< (vz,vxm);
        IVec<SSE, float> vlep = operator< (vz,vxp);
        i = i + vlem + vlep;
        i.store(pr);
    }

    FORCE_INLINE
        //NO_INLINE
        void resolve(const FVec<SSE, double>& vz, const IVec<SSE, float>& bidx, uint32 *pr) const
    {
        const uint32* buckets = reinterpret_cast<const uint32 *>(base_t::data.buckets);
        const double *xi = base_t::data.xi;

        uint32 b1 = buckets[bidx.get1()];
        uint32 b0 = buckets[bidx.get0()];

        const double *p1 = &xi[b1];
        const double *p0 = &xi[b0];

        // read pairs ( X(t-1), X(t) )
        __m128d vx1 = _mm_loadu_pd(p1);
        __m128d vx0 = _mm_loadu_pd(p0);

        // build:
        // { X(t(0)-1), X(t(1)-1) }
        // { X(t(0)),   X(t(1)) }
        __m128d vxm = _mm_shuffle_pd(vx0, vx1, 0);
        __m128d vxp = _mm_shuffle_pd(vx0, vx1, 3);

        IVec<SSE, double> i(b1, b0);
        IVec<SSE, double> vlem = operator< (vz, vxm);
        IVec<SSE, double> vlep = operator< (vz, vxp);
        i = i + vlem + vlep;

        union {
            __m128i vec;
            uint32 ui32[4];
        } u;
        u.vec = i;
        pr[0] = u.ui32[0];
        pr[1] = u.ui32[2];
    }

#ifdef USE_AVX

    FORCE_INLINE
        //NO_INLINE
        void resolve(const FVec<AVX, float>& vz, const IVec<AVX, float>& bidx, uint32 *pr) const
    {
        const uint32* buckets = reinterpret_cast<const uint32 *>(base_t::data.buckets);
        const float *xi = base_t::data.xi;

#if 0   // use gather instructions

        IVec<AVX,float> idxm;
        idxm.setidx(buckets, bidx);
        __m256i z = _mm256_setzero_si256();
        IVec<AVX,float> minusone = _mm256_cmpeq_epi32(z,z);
        IVec<AVX,float> idxp = idxm - minusone;

        FVec<AVX, float> vxm = _mm256_i32gather_ps(xi, idxm, sizeof(float));
        FVec<AVX, float> vxp = _mm256_i32gather_ps(xi, idxp, sizeof(float));
        IVec<AVX, float> ip = idxm;

#else // do not use gather instrucions

        union U {
            __m256i vec;
            uint32 ui32[8];
        } u;

        // read indices t

        const double *p7 = reinterpret_cast<const double *>(&xi[(u.ui32[7] = buckets[bidx.get7()])]);
        const double *p6 = reinterpret_cast<const double *>(&xi[(u.ui32[6] = buckets[bidx.get6()])]);
        const double *p5 = reinterpret_cast<const double *>(&xi[(u.ui32[5] = buckets[bidx.get5()])]);
        const double *p4 = reinterpret_cast<const double *>(&xi[(u.ui32[4] = buckets[bidx.get4()])]);
        const double *p3 = reinterpret_cast<const double *>(&xi[(u.ui32[3] = buckets[bidx.get3()])]);
        const double *p2 = reinterpret_cast<const double *>(&xi[(u.ui32[2] = buckets[bidx.get2()])]);
        const double *p1 = reinterpret_cast<const double *>(&xi[(u.ui32[1] = buckets[bidx.get1()])]);
        const double *p0 = reinterpret_cast<const double *>(&xi[(u.ui32[0] = buckets[bidx.get0()])]);

#if 0 // perform 8 loads in double precision

        // read pairs ( X(t-1), X(t) )
        __m128 xp7 = _mm_castpd_ps(_mm_load_sd(p7));
        __m128 xp6 = _mm_castpd_ps(_mm_load_sd(p6));
        __m128 xp5 = _mm_castpd_ps(_mm_load_sd(p5));
        __m128 xp4 = _mm_castpd_ps(_mm_load_sd(p4));
        __m128 xp3 = _mm_castpd_ps(_mm_load_sd(p3));
        __m128 xp2 = _mm_castpd_ps(_mm_load_sd(p2));
        __m128 xp1 = _mm_castpd_ps(_mm_load_sd(p1));
        __m128 xp0 = _mm_castpd_ps(_mm_load_sd(p0));

        // build:
        // { X(t(0)-1), X(t(1)-1), X(t(2)-1), X(t(3)-1) }
        // { X(t(0)),   X(t(1)),   X(t(2)),   X(t(3)) }
        __m128 h57 = _mm_shuffle_ps(xp5, xp7, (1 << 2) + (1 << 6));  // F- F+ H- H+
        __m128 h46 = _mm_shuffle_ps(xp4, xp6, (1 << 2) + (1 << 6));  // E- E+ G- G+
        __m128 h13 = _mm_shuffle_ps(xp1, xp3, (1 << 2) + (1 << 6));  // B- B+ D- D+
        __m128 h02 = _mm_shuffle_ps(xp0, xp2, (1 << 2) + (1 << 6));  // A- A+ C- C+

        __m128 u01 = _mm_unpacklo_ps(h02, h13);  // A- B- A+ B+
        __m128 u23 = _mm_unpackhi_ps(h02, h13);  // C- D- C+ D+
        __m128 u45 = _mm_unpacklo_ps(h46, h57);  // E- F- E+ F+
        __m128 u67 = _mm_unpackhi_ps(h46, h57);  // G- H- G+ H+

        __m128 abcdm = _mm_shuffle_ps(u01, u23, (0) + (1 << 2) + (0 << 4) + (1 << 6));  // A- B- C- D-
        __m128 abcdp = _mm_shuffle_ps(u01, u23, (2) + (3 << 2) + (2 << 4) + (3 << 6));  // A+ B+ C+ D+
        __m128 efghm = _mm_shuffle_ps(u45, u67, (0) + (1 << 2) + (0 << 4) + (1 << 6));  // E- F- G- H-
        __m128 efghp = _mm_shuffle_ps(u45, u67, (2) + (3 << 2) + (2 << 4) + (3 << 6));  // E+ F+ G+ H+

        FVec<AVX, float> vxp = _mm256_insertf128_ps(_mm256_castps128_ps256(abcdm), efghm, 1);
        FVec<AVX, float> vxm = _mm256_insertf128_ps(_mm256_castps128_ps256(abcdp), efghp, 1);

        IVec<AVX, float> ip(u.vec);

#else   // use __mm256_set_pd

        // read pairs ( X(t-1), X(t) )
        __m256 x0145 = _mm256_castpd_ps(_mm256_set_pd(*p5, *p4, *p1, *p0)); // { x0(t-1), x0(t), x1(t-1), x1(t), x4(t-1), x4(t), x5(t-1), x5(t) }
        __m256 x2367 = _mm256_castpd_ps(_mm256_set_pd(*p7, *p6, *p3, *p2)); // { x2(t-1), x2(t), x3(t-1), x3(t), x6(t-1), x6(t), x7(t-1), x7(t) }

        // { x0(t-1), x1(t-1), x2(t-1), 3(t-1, x4(t-1), x5(t-1), x6(t-1), xt(t-1) }
        FVec<AVX, float> vxm = _mm256_shuffle_ps(x0145, x2367, 0 + (2 << 2) + (0 << 4) + (2 << 6) );
        // { x0(t), x1(t), x2(t), 3(t, x4(t), x5(t), x6(t), xt(t) }
        FVec<AVX, float> vxp = _mm256_shuffle_ps(x0145, x2367, 1 + (3 << 2) + (1 << 4) + (3 << 6) );

        IVec<AVX, float> ip(u.vec);

#endif

#endif

        IVec<AVX, float> vlem = operator< (vz, vxm);
        IVec<AVX, float> vlep = operator< (vz, vxp);
        ip = ip + vlem + vlep;

        ip.store(pr);
    }



    FORCE_INLINE
        //NO_INLINE
        void resolve(const FVec<AVX, double>& vz, const IVec<SSE, float>& bidx, uint32 *pr) const
    {
        union {
            __m256i vec;
            uint64 ui64[4];
        } u;

        const uint32* buckets = reinterpret_cast<const uint32 *>(base_t::data.buckets);
        const double *xi = base_t::data.xi;

        // read indices t
        const double *p3 = &xi[(u.ui64[3] = buckets[bidx.get3()])];
        const double *p2 = &xi[(u.ui64[2] = buckets[bidx.get2()])];
        const double *p1 = &xi[(u.ui64[1] = buckets[bidx.get1()])];
        const double *p0 = &xi[(u.ui64[0] = buckets[bidx.get0()])];

        // read pairs ( X(t-1), X(t) )
        __m128d xp3 = _mm_loadu_pd(p3);
        __m128d xp2 = _mm_loadu_pd(p2);
        __m128d xp1 = _mm_loadu_pd(p1);
        __m128d xp0 = _mm_loadu_pd(p0);

        // build:
        // { X(t(0)-1), X(t(1)-1), X(t(2)-1), X(t(3)-1) }
        // { X(t(0)),   X(t(1)),   X(t(2)),   X(t(3)) }
        __m256d x02 = _mm256_insertf128_pd(_mm256_castpd128_pd256(xp0), xp2, 1);
        __m256d x13 = _mm256_insertf128_pd(_mm256_castpd128_pd256(xp1), xp3, 1);
        FVec<AVX, double> vxm = _mm256_unpacklo_pd(x02,x13);
        FVec<AVX, double> vxp = _mm256_unpackhi_pd(x02,x13);


//        __m128d h01m = _mm_shuffle_pd(xp0, xp1, 0);
//        __m128d h23m = _mm_shuffle_pd(xp2, xp3, 0);
//        __m128d h01p = _mm_shuffle_pd(xp0, xp1, 3);
//        __m128d h23p = _mm_shuffle_pd(xp2, xp3, 3);
//        FVec<AVX, double> vxm = _mm256_insertf128_pd(_mm256_castpd128_pd256(h01m), h23m, 1);
//        FVec<AVX, double> vxp = _mm256_insertf128_pd(_mm256_castpd128_pd256(h01p), h23p, 1);

        IVec<AVX, double> i(u.vec);
        IVec<AVX, double> vlem = operator< (vz,vxm);
        IVec<AVX, double> vlep = operator< (vz,vxp);
        i = i + vlem + vlep;
        i.extractLo32s().store(pr);
    }
#endif

public:

    AlgoVecBase(const T* x, const uint32 n) : base_t(x, n) {}

    void initConstants(Constants& cst) const
    {
        cst.vscaler.setN(base_t::data.scaler);
        cst.vcst0.setN(base_t::data.cst0);
        cst.one.setN(uint32(1));
    }

    void vectorial(uint32 *pr, const T *pz, const Constants& cst) const
    {
        fVec vz(pz);
        resolve(vz, base_t::fun_t::f(cst.vscaler, cst.vcst0, vz), pr);
    }
};
} // namespace Details
} // namespace BinSearch
