package main

import (
	"fmt"
	"math"
)

// Function f(x) = x^2 - 3 = 0
func f(x float64) float64 {
	return math.Pow(x, 2) - 3
}

func main() {
	// Initialize variables as float64
	var lower_bound float64 = 1
	var upper_bound float64 = 2
	var root float64
	var iteration_count int
	var tolerance float64 = 0.000001 // Tolerance to check how close the result is to zero

	// Set up the bisection method
	for {
		middle := (lower_bound + upper_bound) / 2
		potential_root := f(middle)

		// checks the root tolerance
		if math.Abs(potential_root) <= tolerance {
			root = middle
			break
		}

		// Adjust the bounds
		if potential_root > 0 {
			upper_bound = middle
		} else {
			lower_bound = middle
		}
		// Keep count of the iterations
		iteration_count++
	}

	// Output the result
	fmt.Printf("The calculated root is: %.6f\n", root)
	fmt.Printf("The number of iterations it took: %d\n", iteration_count)
}
