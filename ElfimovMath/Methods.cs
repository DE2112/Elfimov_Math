using System;
using static System.Math;

namespace ElfimovMath
{
    public static class Methods
    {
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

                double t;
                var det = Matrix.Determinent(InverseH);
                if (det >= 0d)
                {
                    d = -InverseH * grad;
                    x = prevX + d;
                }
                else
                {
                    d = -grad;

                    t = t0;
                    do
                    {
                        x = prevX + t * d;
                        t /= 2;
                    } while (f(x) - f(prevX) > 0d);
                }
                
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