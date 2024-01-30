#pragma once

#include "Portable.h"

#ifdef USE_SSE42
#ifndef _MSC_VER
#include <popcntintrin.h>
#define popcnt32 _mm_popcnt_u32
#else
#include <intrin.h>
#define popcnt32 __popcnt
#endif
#else // USE_SSE42
namespace BinSearch {
FORCE_INLINE int popcnt32(int x32)
{
    // strictly speaking this is not correct, as it ignores higher order bits
    // however, this is only used on the resuot of movemask on a 128-bit register, which is 8 at most, so it is ok
    // with 256-bit registers, SSE42 is defined, and we do not use this function
    uint8 x = static_cast<uint8>(x32);
    x = (x & 0x55) + (x >> 1 & 0x55);
    x = (x & 0x33) + (x >> 2 & 0x33);
    x = (x & 0x0f) + (x >> 4 & 0x0f);
    return x;
}
} // namespace
#endif

#if defined(USE_AVX) || defined(USE_AVX2)
#include <immintrin.h>
#elif defined __aarch64__
#warning "--- THIS IS AARCH64"
#include <sse2neon.h>
#else
#include <emmintrin.h>
#ifdef USE_SSE41
#include <smmintrin.h>
#endif
#endif

#include "Type.h"

namespace BinSearch {
namespace Details {

template <InstrSet I, class T>
struct FVec;

template <InstrSet I, class T>
struct IVec;

template <InstrSet I, class T>
struct FVec1;

template <> struct InstrIntTraits<SSE>
{
    typedef __m128i vec_t;
};

template <> struct InstrFloatTraits<SSE, float>
{
    typedef __m128  vec_t;
};

template <> struct InstrFloatTraits<SSE, double>
{
    typedef __m128d vec_t;
};

template <InstrSet I, typename T>
struct FTOITraits
{
    typedef IVec<SSE, float> vec_t;
};

#ifdef USE_AVX

template <>
struct FTOITraits<AVX, float>
{
    typedef IVec<AVX, float> vec_t;
};

template <> struct InstrIntTraits<AVX>
{
    typedef __m256i vec_t;
};

template <> struct InstrFloatTraits<AVX, float>
{
    typedef __m256  vec_t;
};

template <> struct InstrFloatTraits<AVX, double>
{
    typedef __m256d vec_t;
};

#endif


template <typename TR>
struct VecStorage
{
    typedef typename TR::vec_t vec_t;

    FORCE_INLINE operator vec_t&() { return vec; }
    FORCE_INLINE operator const vec_t&() const { return vec; }

protected:
    FORCE_INLINE VecStorage() {}
    FORCE_INLINE VecStorage(const vec_t& v) : vec( v ) {}

    vec_t vec;
};

template <InstrSet>
struct IVecBase;

template <>
struct IVecBase<SSE> : VecStorage<InstrIntTraits<SSE>>
{
protected:
    FORCE_INLINE IVecBase() {}
    FORCE_INLINE IVecBase( const vec_t& v) : VecStorage<InstrIntTraits<SSE>>( v ) {}
public:
    FORCE_INLINE static vec_t zero() { return _mm_setzero_si128(); }

    FORCE_INLINE int32 get0() const { return _mm_cvtsi128_si32( vec ); }

    FORCE_INLINE void assignIf( const vec_t& val, const vec_t& mask )
    {
#ifdef USE_SSE41
        vec = _mm_blendv_epi8(vec, val, mask);
#else
        vec = _mm_or_si128(_mm_andnot_si128(mask,vec), _mm_and_si128(mask,val));
#endif
    }
    FORCE_INLINE void orIf(const vec_t& val, const vec_t& mask)
    {
        vec = _mm_or_si128(vec, _mm_and_si128(val,mask));
    }
};

template <>
struct IVec<SSE, float> : IVecBase<SSE>
{
    FORCE_INLINE IVec() {}
    FORCE_INLINE IVec( int32 i ) : IVecBase<SSE>( _mm_set1_epi32( i ) )  {}
    FORCE_INLINE IVec( const vec_t& v) : IVecBase<SSE>( v )              {}
    FORCE_INLINE IVec( uint32 u3, uint32 u2, uint32 u1, uint32 u0) : IVecBase<SSE>( _mm_set_epi32( u3, u2, u1, u0 ) ) {}

    void setN( int32 i ) { vec = _mm_set1_epi32( i ); }

#ifdef USE_SSE41
    FORCE_INLINE int32 get1() const { return _mm_extract_epi32(vec, 1); }
    FORCE_INLINE int32 get2() const { return _mm_extract_epi32(vec, 2); }
    FORCE_INLINE int32 get3() const { return _mm_extract_epi32(vec, 3); }
#else
    FORCE_INLINE int32 get1() const { return _mm_cvtsi128_si32( _mm_shuffle_epi32( vec, 1 ) ); }
    FORCE_INLINE int32 get2() const { return _mm_cvtsi128_si32( _mm_shuffle_epi32( vec, 2 ) ); }
    FORCE_INLINE int32 get3() const { return _mm_cvtsi128_si32( _mm_shuffle_epi32( vec, 3 ) ); }
#endif

    FORCE_INLINE void store( uint32 *pi ) const { _mm_storeu_si128( reinterpret_cast<vec_t*>(pi), vec ); }

    FORCE_INLINE int countbit()
    {
        return popcnt32(_mm_movemask_ps(_mm_castsi128_ps(vec)));
    }
};

template <>
struct IVec<SSE, double> : IVecBase<SSE>
{
    FORCE_INLINE IVec() {}
    FORCE_INLINE IVec( int32 i ) : IVecBase<SSE>( _mm_set1_epi64x( i ) )    {}
    FORCE_INLINE IVec( const vec_t& v) : IVecBase<SSE>( v )                 {}
    FORCE_INLINE IVec( uint64 u1, uint64 u0 ) : IVecBase<SSE>( _mm_set_epi64x(u1, u0) ) {}

    void setN( int32 i ) { vec = _mm_set1_epi64x( i ); }

    FORCE_INLINE int32 get1() const
    {
#ifdef USE_SSE41
        return _mm_extract_epi32(vec, 2);
#else
        return _mm_cvtsi128_si32( _mm_shuffle_epi32( vec, 2 ) );
#endif
    }

    // extract the 2 32 bits integers no. 0, 2 and store them in a __m128i
    FORCE_INLINE IVec<SSE,float> extractLo32s() const
    {
        return _mm_shuffle_epi32(vec, ((2 << 2) | 0));
    }

    FORCE_INLINE void store( uint32 *pi ) const
    {
        pi[0] = get0();
        pi[1] = get1();
    }

    FORCE_INLINE int countbit()
    {
#if 1
        // takes 4 cycles
        __m128i hi = _mm_shuffle_epi32(vec, 2);  // 1 cycle
        __m128i s = _mm_add_epi32(vec, hi);
        int32 x = _mm_cvtsi128_si32(s);
        return -x;
#else
        // takes 6 cycles
        return popcnt32(_mm_movemask_pd(_mm_castsi128_pd(vec)));
#endif
    }
};

template <typename T>
FORCE_INLINE IVec<SSE,T> operator>> (const IVec<SSE,T>& a, unsigned n)            { return _mm_srli_epi32(a, n); }
template <typename T>
FORCE_INLINE IVec<SSE,T> operator<< (const IVec<SSE,T>& a, unsigned n)            { return _mm_slli_epi32(a, n); }
template <typename T>
FORCE_INLINE IVec<SSE,T> operator&  (const IVec<SSE,T>& a, const IVec<SSE,T>& b ) { return _mm_and_si128( a, b ); }
template <typename T>
FORCE_INLINE IVec<SSE,T> operator|  (const IVec<SSE,T>& a, const IVec<SSE,T>& b ) { return _mm_or_si128( a, b ); }
template <typename T>
FORCE_INLINE IVec<SSE,T> operator^  (const IVec<SSE,T>& a, const IVec<SSE,T>& b ) { return _mm_xor_si128( a, b ); }
template <typename T>
FORCE_INLINE IVec<SSE,T> operator+  (const IVec<SSE,T>& a, const IVec<SSE,T>& b ) { return _mm_add_epi32( a, b ); }
template <typename T>
FORCE_INLINE IVec<SSE,T> operator-  (const IVec<SSE,T>& a, const IVec<SSE,T>& b ) { return _mm_sub_epi32( a, b ); }
#ifdef USE_SSE41
template <typename T>
FORCE_INLINE IVec<SSE,T> min        (const IVec<SSE,T>& a, const IVec<SSE,T>& b ) { return _mm_min_epi32( a, b ); }
#endif

typedef VecStorage<InstrFloatTraits<SSE,float>> FVec128Float;

template <>
struct FVec1<SSE, float> : FVec128Float
{
    FORCE_INLINE FVec1() {}
    FORCE_INLINE FVec1( float f ) : FVec128Float( _mm_load_ss( &f ) ) {}
    FORCE_INLINE FVec1( const vec_t& v ): FVec128Float( v ) {}

    FORCE_INLINE float get0() const { return _mm_cvtss_f32( vec ); }
};

template <>
struct FVec<SSE, float> : FVec128Float
{
    FORCE_INLINE FVec() {}
    FORCE_INLINE FVec( float f ) : FVec128Float( _mm_set1_ps( f ) ) {}
    FORCE_INLINE FVec( const float *v ) : FVec128Float( _mm_loadu_ps( v ) ) {}
    FORCE_INLINE FVec( const vec_t& v) : FVec128Float(v) {}
    FORCE_INLINE FVec( float f3, float f2, float f1, float f0 ) : FVec128Float( _mm_set_ps(f3, f2, f1, f0) ) {}

    void set0( float f  ) { vec = _mm_load_ss( &f ); }
    void setN( float f  ) { vec = _mm_set1_ps( f ); }

    FORCE_INLINE void setidx( const float *xi, const IVec<SSE,float>& idx )
    {
        uint32 i0 = idx.get0();
        uint32 i1 = idx.get1();
        uint32 i2 = idx.get2();
        uint32 i3 = idx.get3();
        vec = _mm_set_ps( xi[i3], xi[i2], xi[i1], xi[i0] );
    }

    FORCE_INLINE float get0() const { return _mm_cvtss_f32( vec ); }
    FORCE_INLINE float get1() const { return _mm_cvtss_f32( _mm_shuffle_ps( vec, vec, 1 ) ); }
    FORCE_INLINE float get2() const { return _mm_cvtss_f32( _mm_shuffle_ps( vec, vec, 2 ) ); }
    FORCE_INLINE float get3() const { return _mm_cvtss_f32( _mm_shuffle_ps( vec, vec, 3 ) ); }
};

FORCE_INLINE FVec1<SSE,float> operator+  (const FVec1<SSE,float>& a, const FVec1<SSE,float>& b) { return _mm_add_ss( a, b ); }
FORCE_INLINE FVec1<SSE,float> operator-  (const FVec1<SSE,float>& a, const FVec1<SSE,float>& b) { return _mm_sub_ss( a, b ); }
FORCE_INLINE FVec1<SSE,float> operator*  (const FVec1<SSE,float>& a, const FVec1<SSE,float>& b) { return _mm_mul_ss( a, b ); }
FORCE_INLINE FVec1<SSE,float> operator/  (const FVec1<SSE,float>& a, const FVec1<SSE,float>& b) { return _mm_div_ss( a, b ); }
FORCE_INLINE int              ftoi       (const FVec1<SSE,float>& a)                            { return _mm_cvttss_si32(a); }
FORCE_INLINE IVec<SSE,float> operator>   (const FVec1<SSE,float>& a, const FVec1<SSE,float>& b) { return _mm_castps_si128( _mm_cmpgt_ss( a, b ) ); }
#ifdef USE_FMA
FORCE_INLINE FVec1<SSE, float> mulSub(const FVec1<SSE, float>& a, const FVec1<SSE, float>& b, const FVec1<SSE, float>& c) { return _mm_fmsub_ss(a, b, c); }
#endif

FORCE_INLINE FVec<SSE,float> operator-   (const FVec<SSE,float>& a,  const FVec<SSE,float>& b)  { return _mm_sub_ps( a, b ); }
FORCE_INLINE FVec<SSE,float> operator*   (const FVec<SSE,float>& a,  const FVec<SSE,float>& b)  { return _mm_mul_ps( a, b ); }
FORCE_INLINE FVec<SSE,float> operator/   (const FVec<SSE,float>& a,  const FVec<SSE,float>& b)  { return _mm_div_ps( a, b ); }
FORCE_INLINE IVec<SSE,float> ftoi        (const FVec<SSE,float>& a)                             { return _mm_cvttps_epi32(a); }
FORCE_INLINE IVec<SSE,float> operator<=  (const FVec<SSE,float>& a,  const FVec<SSE,float>& b)  { return _mm_castps_si128( _mm_cmple_ps( a, b ) ); }
FORCE_INLINE IVec<SSE,float> operator>=  (const FVec<SSE,float>& a,  const FVec<SSE,float>& b)  { return _mm_castps_si128( _mm_cmpge_ps( a, b ) ); }
FORCE_INLINE IVec<SSE,float> operator<   (const FVec<SSE,float>& a,  const FVec<SSE,float>& b)  { return _mm_castps_si128(_mm_cmplt_ps(a, b)); }
#ifdef USE_FMA
FORCE_INLINE FVec<SSE, float> mulSub(const FVec<SSE, float>& a, const FVec<SSE, float>& b, const FVec<SSE, float>& c) { return _mm_fmsub_ps(a, b, c); }
#endif

typedef VecStorage<InstrFloatTraits<SSE,double>> FVec128Double;

template <>
struct FVec1<SSE, double> : FVec128Double
{
    FORCE_INLINE FVec1() {}
    FORCE_INLINE FVec1( double f )       : FVec128Double( _mm_load_sd( &f ) ) {}
    FORCE_INLINE FVec1( const vec_t& v ) : FVec128Double( v )                 {}

    FORCE_INLINE double get0() const { return _mm_cvtsd_f64( vec ); }
};

template <>
struct FVec<SSE, double> : FVec128Double
{
    FORCE_INLINE FVec() {}
    FORCE_INLINE FVec( double d )        : FVec128Double( _mm_set1_pd( d ) )   {}
    FORCE_INLINE FVec( const double *v ) : FVec128Double( _mm_loadu_pd( v ) )  {}
    FORCE_INLINE FVec( const vec_t& v)   : FVec128Double( v )                  {}
    FORCE_INLINE FVec( double f1, double f0 ) : FVec128Double( _mm_set_pd(f1, f0) ) {}

    void set0( double f  ) { vec = _mm_load_sd( &f ); }
    void setN( double f  ) { vec = _mm_set1_pd( f ); }

    FORCE_INLINE void setidx( const double *xi, const IVec<SSE,double>& idx )
    {
        vec = _mm_set_pd( xi[idx.get1()], xi[idx.get0()] );
    }

    FORCE_INLINE double get0() const { return _mm_cvtsd_f64( vec ); }
    FORCE_INLINE double get1() const { return _mm_cvtsd_f64( _mm_shuffle_pd( vec, vec, 1 ) ); };
};

FORCE_INLINE FVec1<SSE,double> operator+   (const FVec1<SSE,double>& a, const FVec1<SSE,double>& b) { return _mm_add_sd( a, b ); }
FORCE_INLINE FVec1<SSE,double> operator-   (const FVec1<SSE,double>& a, const FVec1<SSE,double>& b) { return _mm_sub_sd( a, b ); }
FORCE_INLINE FVec1<SSE,double> operator*   (const FVec1<SSE,double>& a, const FVec1<SSE,double>& b) { return _mm_mul_sd( a, b ); }
FORCE_INLINE FVec1<SSE,double> operator/   (const FVec1<SSE,double>& a, const FVec1<SSE,double>& b) { return _mm_div_sd( a, b ); }
FORCE_INLINE int               ftoi        (const FVec1<SSE,double>& a)                             { return _mm_cvttsd_si32(a); }
FORCE_INLINE IVec<SSE,double> operator>    (const FVec1<SSE,double>& a, const FVec1<SSE,double>& b) { return _mm_castpd_si128( _mm_cmpgt_sd( a, b ) ); }
#ifdef USE_FMA
FORCE_INLINE FVec1<SSE, double> mulSub(const FVec1<SSE, double>& a, const FVec1<SSE, double>& b, const FVec1<SSE, double>& c) { return _mm_fmsub_sd(a, b, c); }
#endif

FORCE_INLINE FVec<SSE,double> operator-   (const FVec<SSE,double>& a, const FVec<SSE,double>& b)    { return _mm_sub_pd( a, b ); }
FORCE_INLINE FVec<SSE,double> operator*   (const FVec<SSE,double>& a, const FVec<SSE,double>& b)    { return _mm_mul_pd( a, b ); }
FORCE_INLINE FVec<SSE,double> operator/   (const FVec<SSE,double>& a, const FVec<SSE,double>& b)    { return _mm_div_pd( a, b ); }
FORCE_INLINE IVec<SSE,float>  ftoi        (const FVec<SSE,double>& a)                               { return _mm_cvttpd_epi32(a); }
FORCE_INLINE IVec<SSE,double> operator<=  (const FVec<SSE,double>& a, const FVec<SSE,double>& b)    { return _mm_castpd_si128( _mm_cmple_pd( a, b ) ); }
FORCE_INLINE IVec<SSE,double> operator<   (const FVec<SSE,double>& a, const FVec<SSE,double>& b)    { return _mm_castpd_si128(_mm_cmplt_pd(a, b)); }
FORCE_INLINE IVec<SSE,double> operator>=  (const FVec<SSE,double>& a, const FVec<SSE,double>& b)    { return _mm_castpd_si128( _mm_cmpge_pd( a, b ) ); }
#ifdef USE_FMA
FORCE_INLINE FVec<SSE, double> mulSub(const FVec<SSE, double>& a, const FVec<SSE, double>& b, const FVec<SSE, double>& c ) { return _mm_fmsub_pd(a, b, c); }
#endif

#ifdef USE_AVX

template <>
struct IVecBase<AVX> : VecStorage<InstrIntTraits<AVX>>
{
protected:
    FORCE_INLINE IVecBase() {}
    FORCE_INLINE IVecBase( const vec_t& v) : VecStorage<InstrIntTraits<AVX>>( v ) {}
public:
    FORCE_INLINE static vec_t zero() { return _mm256_setzero_si256(); }

    FORCE_INLINE int32 get0() const { return _mm_cvtsi128_si32(_mm256_castsi256_si128(vec)); }

    FORCE_INLINE void assignIf( const vec_t& val, const vec_t& mask ) { vec = _mm256_blendv_epi8(vec, val, mask); }
    FORCE_INLINE void orIf(const vec_t& val, const vec_t& mask)
    {
        vec = _mm256_blendv_epi8(vec, val, mask);
        //vec = _mm256_or_si256(vec, _mm256_and_si256(val,mask));
    }

    FORCE_INLINE __m128i lo128() const { return _mm256_castsi256_si128(vec); }
    FORCE_INLINE __m128i hi128() const { return _mm256_extractf128_si256(vec, 1); }
};

template <>
struct IVec<AVX, float> : IVecBase<AVX>
{
    FORCE_INLINE IVec() {}
    FORCE_INLINE IVec( int32 i ) : IVecBase<AVX>( _mm256_set1_epi32( i ) )  {}
    FORCE_INLINE IVec( const vec_t& v) : IVecBase<AVX>( v )              {}
    FORCE_INLINE IVec(uint32 u7, uint32 u6, uint32 u5, uint32 u4, uint32 u3, uint32 u2, uint32 u1, uint32 u0) : IVecBase<AVX>(_mm256_set_epi32(u7, u6, u5, u4, u3, u2, u1, u0))          {}

    void setN( int32 i ) { vec = _mm256_set1_epi32( i ); }

    FORCE_INLINE int32 get1() const { return _mm256_extract_epi32(vec, 1); }
    FORCE_INLINE int32 get2() const { return _mm256_extract_epi32(vec, 2); }
    FORCE_INLINE int32 get3() const { return _mm256_extract_epi32(vec, 3); }
    FORCE_INLINE int32 get4() const { return _mm256_extract_epi32(vec, 4); }
    FORCE_INLINE int32 get5() const { return _mm256_extract_epi32(vec, 5); }
    FORCE_INLINE int32 get6() const { return _mm256_extract_epi32(vec, 6); }
    FORCE_INLINE int32 get7() const { return _mm256_extract_epi32(vec, 7); }

    FORCE_INLINE void setidx( const uint32 *bi, const IVec<AVX,float>& idx )
    {
        vec = _mm256_i32gather_epi32(reinterpret_cast<const int32 *>(bi), idx, sizeof(uint32));
    }

    FORCE_INLINE void store( uint32 *pi ) const { _mm256_storeu_si256( reinterpret_cast<vec_t*>(pi), vec ); }

    FORCE_INLINE int countbit()
    {
        return popcnt32(_mm256_movemask_ps(_mm256_castsi256_ps(vec)));
    }
};

template <>
struct IVec<AVX, double> : IVecBase<AVX>
{
    FORCE_INLINE IVec() {}
    FORCE_INLINE IVec( int32 i ) : IVecBase<AVX>( _mm256_set1_epi64x( i ) )    {}
    FORCE_INLINE IVec( const vec_t& v) : IVecBase<AVX>( v )                 {}
    FORCE_INLINE IVec(uint64 u3, uint64 u2, uint64 u1, uint64 u0) : IVecBase<AVX>(_mm256_set_epi64x(u3, u2, u1, u0))          {}

    void setN( int32 i ) { vec = _mm256_set1_epi64x( i ); }

    // extract the 4 32 bits integers no. 0, 2, 4, 6 and store them in a __m128i
    FORCE_INLINE IVec<SSE,float> extractLo32s() const
    {
      union {
        uint32 u32[4];
        __m128i u;
      } mask = {0,2,4,6};
      //__m256 ps256 = _mm256_castsi256_ps(vec);
      //__m128 lo128 = _mm256_castps256_ps128(ps256);
      //__m128 hi128 = _mm256_extractf128_ps(ps256, 1);
      //__m128 blend = _mm_shuffle_ps(lo128, hi128, 0 + (2<<2) + (0<<4) + (2<<6));
      __m256i blend = _mm256_permutevar8x32_epi32(vec, _mm256_castsi128_si256(mask.u));
      return _mm256_castsi256_si128(blend);
    }

    //int32 get1() const { return _mm256_cvtsi256_si32( _mm256_shuffle_epi32( vec, 2 ) ); };
    FORCE_INLINE int32 get1() const { return _mm256_extract_epi32(vec, 2); }

    FORCE_INLINE void store( uint32 *pi ) const
    {
        extractLo32s().store(pi);
    }

    FORCE_INLINE int countbit()
    {
        return popcnt32(_mm256_movemask_pd(_mm256_castsi256_pd(vec)));
    }
};

template <typename T>
FORCE_INLINE IVec<AVX,T> operator>> (const IVec<AVX,T>& a, unsigned n)            { return _mm256_srli_epi32(a, n); }
template <typename T>
FORCE_INLINE IVec<AVX,T> operator<< (const IVec<AVX,T>& a, unsigned n)            { return _mm256_slli_epi32(a, n); }
template <typename T>
FORCE_INLINE IVec<AVX,T> operator&  (const IVec<AVX,T>& a, const IVec<AVX,T>& b ) { return _mm256_and_si256( a, b ); }
template <typename T>
FORCE_INLINE IVec<AVX,T> operator|  (const IVec<AVX,T>& a, const IVec<AVX,T>& b ) { return _mm256_or_si256( a, b ); }
template <typename T>
FORCE_INLINE IVec<AVX,T> operator^  (const IVec<AVX,T>& a, const IVec<AVX,T>& b ) { return _mm256_xor_si256( a, b ); }
template <typename T>
FORCE_INLINE IVec<AVX,T> min        (const IVec<AVX,T>& a, const IVec<AVX,T>& b ) { return _mm256_min_epi32( a, b ); }

FORCE_INLINE IVec<AVX,float> operator+  (const IVec<AVX,float>& a, const IVec<AVX,float>& b ) { return _mm256_add_epi32( a, b ); }
FORCE_INLINE IVec<AVX,float> operator-  (const IVec<AVX,float>& a, const IVec<AVX,float>& b ) { return _mm256_sub_epi32( a, b ); }
FORCE_INLINE IVec<AVX,double> operator+  (const IVec<AVX,double>& a, const IVec<AVX,double>& b ) { return _mm256_add_epi64( a, b ); }
FORCE_INLINE IVec<AVX,double> operator-  (const IVec<AVX,double>& a, const IVec<AVX,double>& b ) { return _mm256_sub_epi64( a, b ); }


typedef VecStorage<InstrFloatTraits<AVX,float>> FVec256Float;

template <>
struct FVec<AVX, float> : FVec256Float
{
    FORCE_INLINE FVec() {}
    FORCE_INLINE FVec( float f ) : FVec256Float( _mm256_set1_ps( f ) ) {}
    FORCE_INLINE FVec( const float *v ) : FVec256Float( _mm256_loadu_ps( v ) ) {}
    FORCE_INLINE FVec( const vec_t& v) : FVec256Float(v) {}
    FORCE_INLINE FVec(float f7, float f6, float f5, float f4, float f3, float f2, float f1, float f0) : FVec256Float(_mm256_set_ps(f7, f6, f5, f4, f3, f2, f1, f0))          {}

    //void set0( float f  ) { vec = _mm256_load_ss( &f ); }
    void setN( float f  ) { vec = _mm256_set1_ps( f ); }

    FORCE_INLINE void setidx( const float *xi, const IVec<AVX,float>& idx )
    {
#if 1 // use gather primitives
        vec = _mm256_i32gather_ps (xi, idx, 4);
#elif 0
        uint32 i0 = idx.get0();
        uint32 i1 = idx.get1();
        uint32 i2 = idx.get2();
        uint32 i3 = idx.get3();
        uint32 i4 = idx.get4();
        uint32 i5 = idx.get5();
        uint32 i6 = idx.get6();
        uint32 i7 = idx.get7();
        vec = _mm256_set_ps( xi[i7], xi[i6], xi[i5], xi[i4], xi[i3], xi[i2], xi[i1], xi[i0] );
#else
        union {
            __m256i vec;
            uint32 ui32[8];
        } i;
        i.vec = static_cast<const __m256i&>(idx);
        vec = _mm256_set_ps(xi[i.ui32[7]], xi[i.ui32[6]], xi[i.ui32[5]], xi[i.ui32[4]], xi[i.ui32[3]], xi[i.ui32[2]], xi[i.ui32[1]], xi[i.ui32[0]]);
#endif
    }

    FORCE_INLINE FVec<SSE, float> lo128() const { return _mm256_castps256_ps128(vec); }
    FORCE_INLINE FVec<SSE, float> hi128() const { return _mm256_extractf128_ps(vec, 1); }

    //FORCE_INLINE float get0() const { return _mm256_cvtss_f32( vec ); }
    //FORCE_INLINE float get1() const { return _mm256_cvtss_f32( _mm256_shuffle_ps( vec, vec, 1 ) ); }
    //FORCE_INLINE float get2() const { return _mm256_cvtss_f32( _mm256_shuffle_ps( vec, vec, 2 ) ); }
    //FORCE_INLINE float get3() const { return _mm256_cvtss_f32( _mm256_shuffle_ps( vec, vec, 3 ) ); }
};

FORCE_INLINE FVec<AVX,float> operator-   (const FVec<AVX,float>& a,  const FVec<AVX,float>& b)  { return _mm256_sub_ps( a, b ); }
FORCE_INLINE FVec<AVX,float> operator*   (const FVec<AVX,float>& a,  const FVec<AVX,float>& b)  { return _mm256_mul_ps( a, b ); }
FORCE_INLINE FVec<AVX,float> operator/   (const FVec<AVX,float>& a,  const FVec<AVX,float>& b)  { return _mm256_div_ps( a, b ); }
FORCE_INLINE IVec<AVX,float> ftoi        (const FVec<AVX,float>& a)                             { return _mm256_cvttps_epi32(a); }
FORCE_INLINE IVec<AVX,float> operator<=  (const FVec<AVX,float>& a,  const FVec<AVX,float>& b)  { return _mm256_castps_si256( _mm256_cmp_ps( a, b, _CMP_LE_OS) ); }
FORCE_INLINE IVec<AVX,float> operator>=  (const FVec<AVX,float>& a,  const FVec<AVX,float>& b)  { return _mm256_castps_si256( _mm256_cmp_ps( a, b, _CMP_GE_OS ) ); }
FORCE_INLINE IVec<AVX,float> operator<   (const FVec<AVX,float>& a,  const FVec<AVX,float>& b)  { return _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_LT_OS )); }
#ifdef USE_FMA
FORCE_INLINE FVec<AVX, float> mulSub(const FVec<AVX, float>& a, const FVec<AVX, float>& b, const FVec<AVX, float>& c) { return _mm256_fmsub_ps(a, b, c); }
#endif

typedef VecStorage<InstrFloatTraits<AVX,double>> FVec256Double;

template <>
struct FVec<AVX, double> : FVec256Double
{
    FORCE_INLINE FVec() {}
    FORCE_INLINE FVec( double d )        : FVec256Double( _mm256_set1_pd( d ) )   {}
    FORCE_INLINE FVec( const double *v ) : FVec256Double( _mm256_loadu_pd( v ) )  {}
    FORCE_INLINE FVec( const vec_t& v)   : FVec256Double( v )                  {}
    FORCE_INLINE FVec(double d3, double d2, double d1, double d0) : FVec256Double(_mm256_set_pd(d3, d2, d1, d0))          {}

    //void set0( double f  ) { vec = _mm256_load_sd( &f ); }
    void setN( double f  ) { vec = _mm256_set1_pd( f ); }

    FORCE_INLINE void setidx( const double *xi, const IVec<SSE,float>& idx )
    {
        vec = _mm256_i32gather_pd(xi, idx, 8);
    }

    FORCE_INLINE void setidx( const double *xi, const IVec<AVX,double>& idx )
    {
        vec = _mm256_i64gather_pd(xi, idx, 8);
    }

//    FORCE_INLINE double get0() const { return _mm256_cvtsd_f64( vec ); }
//    FORCE_INLINE double get1() const { return _mm256_cvtsd_f64( _mm256_shuffle_pd( vec, vec, 1 ) ); };
};

FORCE_INLINE FVec<AVX,double> operator-   (const FVec<AVX,double>& a, const FVec<AVX,double>& b)    { return _mm256_sub_pd( a, b ); }
FORCE_INLINE FVec<AVX,double> operator*   (const FVec<AVX,double>& a, const FVec<AVX,double>& b)    { return _mm256_mul_pd( a, b ); }
FORCE_INLINE FVec<AVX,double> operator/   (const FVec<AVX,double>& a, const FVec<AVX,double>& b)    { return _mm256_div_pd( a, b ); }
FORCE_INLINE IVec<SSE,float>  ftoi        (const FVec<AVX,double>& a)                               { return _mm256_cvttpd_epi32(a); }
FORCE_INLINE IVec<AVX,double> operator<=   (const FVec<AVX,double>& a, const FVec<AVX,double>& b)    { return _mm256_castpd_si256(_mm256_cmp_pd( a, b, _CMP_LE_OS ) ); }
FORCE_INLINE IVec<AVX,double> operator<    (const FVec<AVX,double>& a, const FVec<AVX,double>& b)    { return _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OS)); }
FORCE_INLINE IVec<AVX,double> operator>=   (const FVec<AVX,double>& a, const FVec<AVX,double>& b)    { return _mm256_castpd_si256(_mm256_cmp_pd( a, b, _CMP_GE_OS ) ); }
#ifdef USE_FMA
FORCE_INLINE FVec<AVX, double> mulSub(const FVec<AVX, double>& a, const FVec<AVX, double>& b, const FVec<AVX, double>& c) { return _mm256_fmsub_pd(a, b, c); }
#endif

#endif

} // namepsace Details
} // namespace BinSearch
