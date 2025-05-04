#ifndef THREADS_FIXED_POINT_H
#define THREADS_FIXED_POINT_H

#include <stdint.h>

typedef int f32;

#define F (1 << 14)

inline f32 
to_f32(int n) 
{
    return n * F;
}

inline int
to_int(f32 x)
{
    if (x >= 0)
        return (x + F / 2) / F;
    
    return (x - F / 2) / F;
}

inline f32
add_f32(f32 x, f32 y)
{
    return x + y;
}

inline f32
sub_f32(f32 x, f32 y)
{
    return x - y;
}

inline f32
mul_f32(f32 x, f32 y)
{
    return (int)(((int64_t) x) * y / F);
}

inline f32
div_f32(f32 x, f32 y)
{
    return (int)(((int64_t) x) * F / y);
}

inline f32
add_f32_int(f32 x, int n)
{
    return x + n * F;
}

inline f32
sub_f32_int(f32 x, int n)
{
    return x - n * F;
}

inline f32
mul_f32_int(f32 x, int n)
{
    return x * n;
}

inline f32
div_f32_int(f32 x, int n)
{
    return x / n;
}

#endif // THREADS_FIXED_POINT_H