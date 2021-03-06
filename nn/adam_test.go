package nn

import (
	"math"
	"math/rand"
	"testing"
)

func TestAdam(t *testing.T) {
	t.Parallel()
	dense := NewDense(2, 1)
	euclidean := NewEuclidean(1)

	newDatum := func() ([]float64, []float64) {
		x := make([]float64, 2)
		for i := 0; i < len(x); i++ {
			x[i] = rand.Float64()
		}
		target := make([]float64, 1)
		target[0] = x[0]*math.Pi + x[1]*math.E + math.Phi
		return target, x
	}

	g := dense.Grad()
	optmz := NewAdam(g[:], 1e-3, 0.9, 0.999, 1e-7)
	for i := 0; i < 15000; i++ {
		target, x := newDatum()

		pred := dense.Forward(x)
		euclidean.Forward(target, pred)
		euclidean.Backward()
		dense.Backward(euclidean.Grad()[0][1])

		g := dense.Grad()
		optmz.Update(g[:])
	}

	if math.Abs(dense.W[0]-math.Pi) > 1e-4 {
		t.Fatalf("%+v", dense)
	}
	if math.Abs(dense.W[1]-math.E) > 1e-4 {
		t.Fatalf("%+v", dense)
	}
	if math.Abs(dense.B[0]-math.Phi) > 1e-4 {
		t.Fatalf("%+v", dense)
	}
}
