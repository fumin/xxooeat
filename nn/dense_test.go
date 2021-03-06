package nn

import (
	"flag"
	"log"
	"math"
	"os"
	"reflect"
	"testing"
)

func TestDenseForward(t *testing.T) {
	t.Parallel()
	model := NewDense(3, 2)
	copy(model.w, []float64{1, 2, 3, 4, 5, 6})
	copy(model.b, []float64{0.1, 0.2})
	x := []float64{10, 20, 30}

	y := model.Forward(x)
	if !reflect.DeepEqual([]float64{140.1, 320.2}, y) {
		t.Fatalf("%+v", y)
	}
}

func TestDenseBackwardX(t *testing.T) {
	t.Parallel()
	model := NewDense(3, 2)
	w := []float64{1, 2, 3, 4, 5, 6}
	copy(model.w, w)
	b := []float64{0.1, 0.2}
	copy(model.b, b)
	x := []float64{10, 20, 30}

	for i := 0; i < len(x); i++ {
		y := sum(model.Forward(x))

		xx := make([]float64, len(x))
		copy(xx, x)
		xx[i] += 1e-6
		yy := sum(model.Forward(xx))

		g := (yy - y) / (xx[i] - x[i])

		model.Forward(x)
		model.Backward([]float64{1, 1})
		grad := model.Grad()
		if math.Abs(g-grad[0][1][i]) > 1e-6 {
			t.Fatalf("%d %f %+v", i, g, grad[0][1][i])
		}
	}
}

func TestDenseBackwardW(t *testing.T) {
	t.Parallel()
	model := NewDense(3, 2)
	w := []float64{1, 2, 3, 4, 5, 6}
	b := []float64{0.1, 0.2}
	copy(model.b, b)
	x := []float64{10, 20, 30}

	for i := 0; i < len(model.w); i++ {
		copy(model.w, w)
		y := sum(model.Forward(x))

		model.w[i] += 1e-6
		yy := sum(model.Forward(x))

		g := (yy - y) / (model.w[i] - w[i])

		model.Forward(x)
		model.Backward([]float64{1, 1})
		grad := model.Grad()
		if math.Abs(g-grad[1][1][i]) > 1e-6 {
			t.Fatalf("%d %f %+v", i, g, grad[1][1][i])
		}
	}
}

func TestDenseBackwardB(t *testing.T) {
	t.Parallel()
	model := NewDense(3, 2)
	w := []float64{1, 2, 3, 4, 5, 6}
	copy(model.w, w)
	b := []float64{0.1, 0.2}
	x := []float64{10, 20, 30}

	for i := 0; i < len(model.b); i++ {
		copy(model.b, b)
		y := sum(model.Forward(x))

		model.b[i] += 1e-6
		yy := sum(model.Forward(x))

		g := (yy - y) / (model.b[i] - b[i])

		model.Forward(x)
		model.Backward([]float64{1, 1})
		grad := model.Grad()
		if math.Abs(g-grad[2][1][i]) > 1e-6 {
			t.Fatalf("%d %f %+v", i, g, grad[2][1][i])
		}
	}
}

func sum(x []float64) float64 {
	var y float64 = 0
	for i := 0; i < len(x); i++ {
		y += x[i]
	}
	return y
}

func TestMain(m *testing.M) {
	flag.Parse()
	log.SetFlags(log.LstdFlags | log.Lmicroseconds | log.Lshortfile)
	os.Exit(m.Run())
}
