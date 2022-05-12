using System;
using System.Collections.Generic;
using static System.Math;

namespace ElfimovMath
{
    public static class Methods
    {
        public static double UniformSearch(Func<double, double> f, double a, double b, double eps)
        {
            var n = (b - a) / eps;
            var minY = f(a);
            var minX = a;

            for (int i = 1; i <= n; i++)
            {
                var x = a + i * (b - a) / n;
                var y = f(x);
                if (minY > y)
                {
                    minY = y;
                    minX = x;
                }
            }

            return minX;
        }
        
        public static Vector GradientDescent(Func<Vector, double> f, Vector x0, double eps1, double eps2, double M, double t0, out int k)
        {
            Vector grad = f.Gradient(x0);
            var x = x0.Clone();
            var prevX = x.Clone();

            k = 0;
            var isMatching = false;
            while (grad.Norm > eps1 && k < M)
            {
                grad = f.Gradient(x);

                var t = t0;
                do
                {
                    x = prevX - t * grad;
                    t /= 2;
                } while (f(x) - f(prevX) > 0d);

                if ((x - prevX).Norm < eps2 && Abs(f(x) - f(prevX)) < eps2)
                {
                    if (isMatching)
                    {
                        return x;
                    }

                    isMatching = true;
                }
                else
                {
                    isMatching = false;
                }
                
                prevX = x.Clone();
                k++;
            }

            return x;
        }
        
        public static Vector FastGradientDescent(Func<Vector, double> f, Vector x0, double eps1, double eps2, double M, out int k)
        {
            Vector grad = f.Gradient(x0);
            var x = x0.Clone();
            var prevX = x.Clone();

            k = 0;
            var isMatching = false;
            while (grad.Norm > eps1 && k < M)
            {
                grad = f.Gradient(x);

                var tk = Round(UniformSearch(t => f(prevX - t * grad), 0d, 1d, eps1), 4);
                x = prevX - tk * grad;

                if ((x - prevX).Norm < eps2 && Abs(f(x) - f(prevX)) < eps2)
                {
                    if (isMatching)
                    {
                        return x;
                    }

                    isMatching = true;
                }
                else
                {
                    isMatching = false;
                }
                
                prevX = x.Clone();
                k++;
            }

            return x;
        }

        public static Vector FletcherReeves(Func<Vector, double> f, Vector x0, double eps1, double eps2, int M,
            double t0, out int k)
        {
            var grad = f.Gradient(x0);
            var prevGrad = grad.Clone();
            var x = x0.Clone(); 
            var prevX = x.Clone(); 
            Vector d = new Vector(grad.Size);
            
            k = 0; 
            var isMatching = false;
            while (grad.Norm > eps1 && k < M) 
            {
                grad = f.Gradient(x);
                var beta = Pow(grad.Norm, 2) / Pow(prevGrad.Norm, 2);
        
                if (k != 0)
                {
                    d = -grad + beta * d;
                }
                else
                {
                    d = -grad;
                }
        
                var t = t0;
                do
                {
                    x = prevX + t * d;
                    t /= 2;
                } while (f(x) - f(prevX) > 0d);
        
                if ((x - prevX).Norm < eps2 && Abs(f(x) - f(prevX)) < eps2)
                {
                    if (isMatching)
                    {
                        return x;
                    }
        
                    isMatching = true;
                }
                else
                {
                    isMatching = false;
                }
        
                prevX = x.Clone();
                prevGrad = grad.Clone();
                k++;
            }
            
            return x;
        }

        public static Vector Newton(Func<Vector, double> f, Vector x0, double eps1, double eps2, double M, double t0,
            out int k)
        {
            var grad = f.Gradient(x0);
            var x = x0.Clone();
            var H = f.HessianMatrix(x0);
            var prevX = x.Clone(); 
            Vector d = new Vector(grad.Size);
            
            k = 0; 
            var isMatching = false;
            while (grad.Norm > eps1 && k < M) 
            {
                grad = f.Gradient(x);
                H = f.HessianMatrix(x);
                var InverseH = H.Inverse();

                if (InverseH.IsPositive())
                {
                    d = -InverseH * grad;
                    x = prevX + d;
                }
                else
                {
                    d = -grad;
                }
                
                var t = t0;
                do
                {
                    x = prevX + t * d;
                    t /= 2;
                } while (f(x) - f(prevX) > 0d);
                
                if ((x - prevX).Norm < eps2 && Abs(f(x) - f(prevX)) < eps2)
                {
                    if (isMatching)
                    {
                        return x;
                    }
        
                    isMatching = true;
                }
                else
                {
                    isMatching = false;
                }
        
                prevX = x.Clone();
                k++;
            }
            
            return x;
        }
        
        public static Vector NewtonRaphson(Func<Vector, double> f, Vector x0, double eps1, double eps2, double M,
            out int k)
        {
            var grad = f.Gradient(x0);
            var x = x0.Clone();
            var H = f.HessianMatrix(x0);
            var prevX = x.Clone(); 
            Vector d = new Vector(grad.Size);
            
            k = 0; 
            var isMatching = false;
            while (grad.Norm > eps1 && k < M) 
            {
                grad = f.Gradient(x);
                H = f.HessianMatrix(x);
                var InverseH = H.Inverse();
                
                if (InverseH.IsPositive())
                {
                    d = -InverseH * grad;
                }
                else
                {
                    d = -grad;
                }
                
                var tk = Round(UniformSearch(t => f(prevX + t * d), 0d, 1d, eps1), 4);
                x = prevX + tk * d;
                
                if ((x - prevX).Norm < eps2 && Abs(f(x) - f(prevX)) < eps2)
                {
                    if (isMatching)
                    {
                        return x;
                    }
        
                    isMatching = true;
                }
                else
                {
                    isMatching = false;
                }
        
                prevX = x.Clone();
                k++;
            }
            
            return x;
        }
    }
}