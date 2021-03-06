package nn

import (
	"math"
	"testing"
)

func TestTanhForward(t *testing.T) {
	t.Parallel()
	model := NewTanh(2)
	x := []float64{0.5, 1}
	y := model.Forward(x)
	if math.Abs(math.Tanh(0.5)-y[0]) > 1e-6 {
		t.Fatalf("%+v", y)
	}
	if math.Abs(math.Tanh(1)-y[1]) > 1e-6 {
		t.Fatalf("%+v", y)
	}
}

func TestTanhBackward(t *testing.T) {
	t.Parallel()
	model := NewTanh(2)
	x := []float64{0.5, 1}

	for i := 0; i < len(x); i++ {
		y := sum(model.Forward(x))

		xx := make([]float64, len(x))
		copy(xx, x)
		xx[i] += 1e-6
		yy := sum(model.Forward(xx))

		g := (yy - y) / (xx[i] - x[i])

		model.Forward(x)
		model.Backward([]float64{2, 2})
		grad := model.Grad()
		if math.Abs(2*g-grad[0][1][i]) > 1e-6 {
			t.Fatalf("%d %f %+v", i, g, grad)
		}
	}
}
