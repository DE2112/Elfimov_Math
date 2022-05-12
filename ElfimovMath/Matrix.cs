using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ElfimovMath
{
    public class Matrix
    {
        private double[,] _elements;
        public int Size => _elements.GetLength(0);

        public Matrix(int size)
        {
            _elements = new double[size, size];
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    _elements[i, j] = 0d;
                }
            }
        }
        
        public Matrix(double[,] elements)
        {
            _elements = elements.Clone() as double[,];
        }

        public double this[int row, int col]
        {
            get => _elements[row,col];
            set => _elements[row,col] = value;
        }

        public static Matrix Input(int size)
        {
            var elements = new double[size,size];
            for (int i = 0; i < size; i++)
            {
                var row = Array.ConvertAll(Console.ReadLine()?.Split(' '), x => double.Parse(x));
                for (int j = 0; j < size; j++)
                {
                    elements[i, j] = row[j];
                }
            }

            return new Matrix(elements);
        }
        
        public static Matrix Input(StreamReader file, int size)
        {
            var elements = new double[size,size];
            for (int i = 0; i < size; i++)
            {
                var row = Array.ConvertAll(file.ReadLine()?.Split(' '), x => double.Parse(x));
                for (int j = 0; j < size; j++)
                {
                    elements[i, j] = row[j];
                }
            }

            return new Matrix(elements);
        }

        public static Matrix IdentityMatrix(int size)
        {
            var elements = new double[size,size];
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    elements[i,j] = i == j ? 1d : 0d;
                }
            }

            return new Matrix(elements);
        }

        public Matrix Clone()
        {
            return new Matrix(_elements);
        }
        
        public static Matrix operator *(double c, Matrix a)
        {
            var elements = new double[a.Size,a.Size];
            for (int i = 0; i < a.Size; i++)
            {
                for (int j = 0; j < a.Size; j++)
                {
                    elements[i, j] = c * a[i, j];
                }
            }
            
            return new Matrix(elements);
        }
        
        public static Matrix operator +(Matrix a, Matrix b)
        {
            if (a.Size == b.Size)
            {
                var elements = new double[a.Size,a.Size];
                for (int i = 0; i < a.Size; i++)
                {
                    for (int j = 0; j < a.Size; j++)
                    {
                        elements[i, j] = a[i, j] + b[i, j];
                    }
                }
                
                return new Matrix(elements);
            }

            return null;
        }
        
        public static Matrix operator -(Matrix a)
        {
            return -1f * a;
        }

        public static Matrix operator -(Matrix a, Matrix b)
        {
            return a + (-b);
        }

        public static Vector operator *(Matrix a, Vector x)
        {
            if (a.Size == x.Size)
            {
                var elements = new double[a.Size];

                for (int i = 0; i < a.Size; i++)
                {
                    var element = 0d;

                    for (int j = 0; j < a.Size; j++)
                    {
                        element += a[i, j] * x[i];
                    }
                    elements[i] = element;
                }

                return new Vector(elements);
            }

            return null;
        }
        
        public static Matrix operator /(Matrix a, double c)
        {
            return 1 / c * a;
        }

        public void InterchangeRows(int i, int j)
        {
            for (int k = 0; k < Size; k++)
            {
                (_elements[i, k], _elements[j, k]) = (_elements[j, k], _elements[i, k]);
            }
        }
        
        public static Matrix Minor(Matrix matrix, int iRow, int iCol)
        {
            var minor = new Matrix(matrix.Size - 1);
            var m = 0;
            var n = 0;

            for (int i = 0; i < matrix.Size; i++)
            {
                if (i == iRow)
                    continue;
                n = 0;
                for (int j = 0; j < matrix.Size; j++)
                {
                    if (j == iCol)
                        continue;
                    minor[m, n] = matrix[i, j];
                    n++;	
                }
                m++;
            }
            return minor;
        }

        public Matrix PrincipalMinor(int size)
        {
            var elements = new double[size, size];

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    elements[i, j] = this[i, j];
                }
            }

            return new Matrix(elements);
        }

        public static double Determinent(Matrix matrix)
        {
            var det = 0d;
            var a = matrix.Clone();
            var size = a.Size;
            if (size == 1) return matrix[0,0];
            for (int j = 0; j < size; j++)
            {
                det += (matrix[0, j] * Determinent(Minor(matrix, 0, j)) * Math.Pow(-1, 0 + j));
            }
            return det;
        }

        public double Determinent()
        {
            return Determinent(this);
        }
        
        public Matrix Inverse()
        {
            var det = Matrix.Determinent(this);
            if (Math.Abs(det) != 0d) 
            {
                return Adjoint() / det;
            }

            return null;
        }

        public Matrix Adjoint()
        {
            Matrix adjointMatrix = new Matrix(Size);
            for (int i = 0; i < Size; i++)
            {
                for (int j = 0; j < Size; j++)
                {
                    adjointMatrix[i, j] = Math.Pow(-1, i + j) * Minor(this, i, j).Determinent();
                }
            }

            adjointMatrix = adjointMatrix.Transpose();
            return adjointMatrix;
        }
        
        public Matrix Transpose()
        {
            var transposeMatrix = new Matrix(Size);
            for (int i = 0; i < Size; i++)
            {
                for (int j = 0; j < Size; j++)
                {
                    transposeMatrix[i, j] = this[j, i];
                }
            }
            return transposeMatrix;
        }

        public bool IsPositive()
        {
            for (int i = 1; i <= this.Size; i++)
            {
                if (Determinent(this.PrincipalMinor(i)) < 0d) return false;
            }
            
            return true;
        }

        public override string ToString()
        {
            var value = "";
            for (int i = 0; i < Size; i++)
            {
                for (int j = 0; j < Size; j++)
                {
                    value += $"{Math.Round(_elements[i, j], 4)} ";
                }
                value += "\n";
            }
            return value;
        }
    }
}
