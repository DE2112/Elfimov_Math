using System;

namespace ElfimovMath
{
    public static class Operations
    {
        private static double DELTA = 0.00001d;
        
        public static double FirstPartialDerivative(this Func<Vector, double> f, Vector x, int i)
        {
            var xPlusDelta = x.Clone();
            xPlusDelta[i] += DELTA;

            return (f(xPlusDelta) - f(x)) / DELTA;
        }

        public static double SecondPartialDerivative(this Func<Vector, double> f, Vector x, int first, int second)
        {
            var x1 = x.CloneAndAddAt(DELTA, first).CloneAndAddAt(DELTA, second);
            var x2 = x.CloneAndAddAt(DELTA, first).CloneAndAddAt(-DELTA, second);
            var x3 = x.CloneAndAddAt(-DELTA, first).CloneAndAddAt(DELTA, second);
            var x4 = x.CloneAndAddAt(-DELTA, first).CloneAndAddAt(-DELTA, second);

            return (f(x1) - f(x2) - f(x3) + f(x4)) / (4d * DELTA * DELTA);
        }

        public static Vector Gradient(this Func<Vector, double> f, Vector x)
        {
            var grad = new Vector(x.Size);
            for (int i = 0; i < grad.Size; i++)
            {
                grad[i] = f.FirstPartialDerivative(x, i);
            }

            return grad;
        }

        public static Matrix HessianMatrix(this Func<Vector, double> f, Vector x)
        {
            var elements = new double[x.Size, x.Size];

            for (int i = 0; i < x.Size; i++)
            {
                for (int j = 0; j < x.Size; j++)
                {
                    elements[i, j] = SecondPartialDerivative(f, x, i, j);
                }
            }

            return new Matrix(elements);
        }
    }
}