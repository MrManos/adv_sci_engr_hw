package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// Function that accepts a real matrix and prints it
func PrintRealMatrix(m mat.Matrix) {
	fmt.Printf("%v\n", mat.Formatted(m, mat.Prefix(""), mat.Excerpt(0)))
}

// Function to print complex matrices (for mat.CDense)
func PrintComplexMatrix(m *mat.CDense) {
	r, c := m.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			fmt.Printf("%6.2f ", m.At(i, j))
		}
		fmt.Println()
	}
}

func main() {
	// Define a 2x2 matrix (square matrix)
	data := []float64{
		4, -2,
		1, 1,
	}
	matrix := mat.NewDense(2, 2, data)

	fmt.Println("Matrix A:")
	PrintRealMatrix(matrix)

	// Create an Eigen decomposition object
	var eig mat.Eigen

	// Compute the eigenvalue decomposition of the matrix
	ok := eig.Factorize(matrix, mat.EigenBoth) // Compute both eigenvalues and eigenvectors
	if !ok {
		fmt.Println("Eigenvalue decomposition failed")
		return
	}

	// Get and print the eigenvalues (complex)
	eigenvalues := eig.Values(nil)
	fmt.Printf("Eigenvalues: %v\n", eigenvalues)

	// Extract and print the eigenvectors (stored in a complex matrix)
	var eigenvectors mat.CDense
	eig.VectorsTo(&eigenvectors)

	fmt.Println("Eigenvectors:")
	PrintComplexMatrix(&eigenvectors)
}
