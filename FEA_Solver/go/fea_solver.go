package main

import (
	"fmt"
	"log"
	"math"

	"gonum.org/v1/gonum/mat"
)

// Function to print a matrix in a readable format
func PrintMatrix(m mat.Matrix) {
	fmt.Printf("%v\n", mat.Formatted(m, mat.Prefix(""), mat.Excerpt(0)))
}

// svdCalculator computes the SVD of a matrix, its condition number, and its inverse if invertible
func svdCalculator(A *mat.Dense) (U, S, V mat.Dense, conditionNumber float64, AInv *mat.Dense, err error) {
	var svd mat.SVD
	if ok := svd.Factorize(A, mat.SVDThin); !ok {
		return U, S, V, 0, nil, fmt.Errorf("SVD factorization failed")
	}

	// Retrieve U, S (singular values), and V matrices from the SVD
	U = *svd.UTo(&U)
	V = *svd.VTo(&V)
	singularValues := svd.Values(nil)

	// Check for invertibility by ensuring all singular values are non-zero
	minSingular, maxSingular := math.Inf(1), math.Inf(-1)
	for _, s := range singularValues {
		if s == 0 {
			return U, S, V, 0, nil, fmt.Errorf("matrix is non-invertible")
		}
		if s < minSingular {
			minSingular = s
		}
		if s > maxSingular {
			maxSingular = s
		}
	}

	// Condition number: ratio of the largest to smallest singular values
	conditionNumber = maxSingular / minSingular

	// Calculate the inverse: A^{-1} = V * S^{-1} * U^T
	SInv := mat.NewDiagDense(len(singularValues), nil)
	for i, s := range singularValues {
		SInv.SetDiag(i, 1/s)
	}
	var U_T mat.Dense
	U_T.CloneFrom(U.T())
	var temp mat.Dense
	temp.Mul(SInv, &U_T)
	AInv = new(mat.Dense)
	AInv.Mul(&V, &temp)

	return U, *SInv, V, conditionNumber, AInv, nil
}

func main() {
	// Define a 2x2 matrix
	data := []float64{1, 2, 3, 4}
	matrix := mat.NewDense(2, 2, data)

	// Pass the matrix to the svdCalculator function
	U, S, V, conditionNumber, AInv, err := svdCalculator(matrix)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("U matrix:")
	PrintMatrix(&U)
	fmt.Println("S matrix (Inverse of singular values):")
	PrintMatrix(&S)
	fmt.Println("V matrix:")
	PrintMatrix(&V)
	fmt.Printf("Condition number: %f\n", conditionNumber)

	if AInv != nil {
		fmt.Println("Inverse matrix:")
		PrintMatrix(AInv)
	} else {
		fmt.Println("Matrix is non-invertible, so no inverse is returned.")
	}
}
