using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ElfimovMath
{
    public class Vector
    {
        private double[] _elements;

        public int Size => _elements.Length;

        public double Norm
        {
            get
            {
                var elementSquareSum = 0d;
            
                foreach (var element in _elements)
                {
                    elementSquareSum += Math.Pow(element, 2);
                }
            
                return Math.Sqrt(elementSquareSum);
            }
        }

        public Vector(double[] elements)
        {
            _elements = elements.Clone() as double[];
        }
        
        public Vector(int size)
        {
            _elements = new double[size];

            for (int i = 0; i < size; i++)
            {
                _elements[i] = 0d;
            }
        }

        public double this[int i]
        {
            get => _elements[i];
            set => _elements[i] = value;
        }
        
        public static Vector Input()
        {
            var elements = Array.ConvertAll(Console.ReadLine()?.Split(' '), x => double.Parse(x));
            return new Vector(elements);
        }
        
        public static Vector Input(StreamReader file)
        {
            var elements = Array.ConvertAll(file.ReadLine()?.Split(' '), x => double.Parse(x));
            return new Vector(elements);
        }
        
        public static Vector One(int size)
        {
            var elements = new double[size];

            for (int i = 0; i < size; i++)
            {
                elements[i] = 1d;
            }

            return new Vector(elements);
        }
        
        public Vector Clone()
        {
            return new Vector(_elements);
        }

        public Vector CloneAndAddAt(double value, int i)
        {
            var elements = _elements.Clone() as double[];
            elements[i] += value;

            return new Vector(elements);
        }
        
        public static Vector operator *(double c, Vector a)
        {
            var elements = new double[a.Size];
            for (int i = 0; i < elements.Length; i++)
            {
                
                elements[i] = c * a[i];
            }

            return new Vector(elements);
        }
        
        public static Vector operator *(Vector a, double c)
        {
            return c * a;
        }

        public static Vector operator /(Vector a, double c)
        {
            return 1 / c * a;
        }
        
        public static Vector operator +(Vector a, Vector b)
        {
            if (a.Size == b.Size)
            {
                var elements = new double[a.Size];
                for (int i = 0; i < elements.Length; i++)
                {
                    elements[i] = a[i] + b[i];
                }

                return new Vector(elements);
            }

            return null;
        }

        public static Vector operator -(Vector v)
        {
            return -1d * v;
        }

        public static Vector operator -(Vector a, Vector b)
        {
            return a + -1d * b;
        }

        public static double operator *(Vector a, Vector b)
        {
            if (a.Size == b.Size)
            {
                var sum = 0d;
                for (int i = 0; i < a.Size; i++)
                {
                    sum += a[i] * b[i];
                }

                return sum;
            }

            return 0d;
        }

        public double Avg()
        {
            var sum = 0d;
            foreach (var value in _elements)
            {
                sum += value;
            }

            return sum / Size;
        }

        public double AbsMin()
        {
            var min = Math.Abs(_elements[0]);
            foreach (var value in _elements)
            {
                if (Math.Abs(value) < min) min = Math.Abs(value);
            }

            return min;
        }

        public void Replace(int i, int j)
        {
            (_elements[i], _elements[j]) = (_elements[j], _elements[i]);
        }
        
        public override string ToString()
        {
            var value = "(";

            for (int i = 0; i < Size - 1; i++)
            {
                value += $"{Math.Round(_elements[i], 4)}, ";
            }
            value += $"{Math.Round(_elements[Size-1], 4)})";
            
            return value;
        }
    }
}