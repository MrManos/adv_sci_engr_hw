package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
)

// Function that accepts a matrix and prints it
func PrintMatrix(m mat.Matrix) {
	fmt.Printf("%v\n", mat.Formatted(m, mat.Prefix(""), mat.Excerpt(0)))
}

// A is an arbitrary matrix of size m x n
// Pass by refrence to save memory
func svd_calculator(A *mat.Dense) {
	// This creates an empty matrix
	var A_T_A mat.Dense

	// Finds the product of A^TA
	A_T_A.Mul(A.T(), A)

	// Creates an Empty Eigen matrix
	var eig mat.Eigen

	// Computes both the eigenvectors and eigenvalues
	eig.Factorize( &A_T_A, mat.EigenBoth)

	if (eig == nil) {
		fmt.Println("Your matrix is most likely non-invertable")
		panic("Matrix is invertable")
	}




}

func main() {
	// Create a 2x2 matrix
	data := []float64{1, 2, 3, 4}
	matrix := mat.NewDense(2, 2, data)

	// Pass the matrix to the function
	PrintMatrix(matrix)
}
