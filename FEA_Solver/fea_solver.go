package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"

	"log"

)

// Function that accepts a matrix and prints it
func PrintMatrix(m mat.Matrix) {
	fmt.Printf("%v\n", mat.Formatted(m, mat.Prefix(""), mat.Excerpt(0)))
}

// A is an arbitrary matrix os size m x n
func svd_calculator(A *mat.Dense){
	// This holds a matrix 
	var invA mat.Dense

	// Finds the inverse of matrix A 
	inverseA := invA.Inverse(A)

	A_T_A := inverseA * A mat.Matrix

	// Creates an Eigen matrix 
	var eig mat.Eigen
	// Computes both the eigenvectors and eigenvalues
	ok := eig.Factorize(A_T_A, mat.EigenBoth)

	values := eig.Values(nil)
}




func main() {
	// Create a 2x2 matrix
	data := []float64{1, 2, 3, 4}
	matrix := mat.NewDense(2, 2, data)

	// Pass the matrix to the function
	PrintMatrix(matrix)
}