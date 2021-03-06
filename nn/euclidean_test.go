package nn

import (
	"math"
	"testing"
)

func TestEuclideanForward(t *testing.T) {
	t.Parallel()
	model := NewEuclidean(2)
	x := []float64{1, 2}
	target := []float64{3, 5}
	y := model.Forward(target, x)
	if 13 != y {
		t.Fatalf("%f", y)
	}
}

func TestEuclideanBackward(t *testing.T) {
	t.Parallel()
	model := NewEuclidean(2)
	x := []float64{1, 2}
	target := []float64{3, 5}

	for i := 0; i < len(x); i++ {
		y := model.Forward(target, x)

		xx := make([]float64, len(x))
		copy(xx, x)
		xx[i] += 1e-6
		yy := model.Forward(target, xx)

		g := (yy - y) / (xx[i] - x[i])

		model.Forward(target, x)
		model.Backward()
		grad := model.Grad()
		if math.Abs(g-grad[0][1][i]) > 1e-5 {
			t.Fatalf("%d %f %+v", i, g, grad)
		}
	}
}
