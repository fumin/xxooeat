package nn

import (
	"math"
	"testing"
)

func TestSumForward(t *testing.T) {
	t.Parallel()
	model := NewSum(2)
	x := []float64{1, 2}
	y := model.Forward(x)
	if 3 != y[0] {
		t.Fatalf("%f", y)
	}
}

func TestSumBackward(t *testing.T) {
	t.Parallel()
	model := NewSum(2)
	x := []float64{1, 2}

	for i := 0; i < len(x); i++ {
		y := model.Forward(x)[0]

		xx := make([]float64, len(x))
		copy(xx, x)
		xx[i] += 1e-6
		yy := model.Forward(xx)[0]

		g := (yy - y) / (xx[i] - x[i])

		model.Forward(x)
		model.Backward([]float64{2})
		grad := model.Grad()

		if math.Abs(2*g-grad[0][1][i]) > 1e-6 {
			t.Fatalf("%d %f %+v", i, g, grad)
		}
	}
}
