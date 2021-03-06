package nn

import (
	"log"
	"math"
	"math/rand"
	"testing"
)

func TestLogisticRegression(t *testing.T) {
	t.Parallel()
	model := NewModel(2, 3)
	x := []float64{0, 1}
	target := []float64{0}
	if b(x[0]) || b(x[1]) {
		target[0] = 1
	}

	for i := 3; i >= 0; i-- {
		model.Forward(target, x)
		w := model.Backward()[i][0]
		oriW := make([]float64, len(w))
		copy(oriW, w)
		for j := 0; j < len(w); j++ {
			y, _ := model.Forward(target, x)

			w[j] += 1e-6
			yy, _ := model.Forward(target, x)

			g := (yy - y) / (w[j] - oriW[j])

			copy(w, oriW)
			model.Forward(target, x)
			grad := model.Backward()
			if math.Abs(g-grad[i][1][j]) > 1e-5 {
				t.Fatalf("weightID: %d, j: %d, trueG: %f, computedG: %+v", i, j, g, grad[i][1][j])
			}
		}
	}
}

func TestLearn(t *testing.T) {
	t.Parallel()
	x := make([]float64, 4)
	model := NewModel(len(x), 3)

	newDatum := func() ([]float64, []float64) {
		for j := 0; j < len(x); j++ {
			x[j] = float64(rand.Intn(2))
		}
		targetBool := (b(x[0]) || b(x[1])) && (b(x[2]) != b(x[3]))
		target := []float64{0}
		if targetBool {
			target[0] = 1
		}
		return target, x
	}

	// Train.
	learningRate := 1e-1
	for i := 0; i < 20000; i++ {
		target, x := newDatum()
		model.Forward(target, x)
		gradients := model.Backward()
		for j := 0; j < len(gradients); j++ {
			weights := gradients[j][0]
			grads := gradients[j][1]
			for k := 0; k < len(weights); k++ {
				weights[k] += -learningRate * grads[k]
			}
		}
	}

	// Test.
	trials := make([]float64, 1000)
	for i := 0; i < len(trials); i++ {
		target, x := newDatum()
		_, predF := model.Forward(target, x)
		var pred float64 = 0
		if predF > 0.5 {
			pred = 1
		}

		if pred == target[0] {
			trials[i] = 1
		}
	}
	accuracy := sum(trials) / float64(len(trials))
	if accuracy != 1 {
		t.Fatalf("%f", accuracy)
	}
}

func b(x float64) bool {
	if int(x) == 0 {
		return false
	}
	return true
}

type Model struct {
	dense0    *Dense
	logistic0 *Logistic
	pool      *Dense
	loss      *Euclidean
}

func NewModel(x, h int) *Model {
	model := &Model{}
	model.dense0 = NewDense(x, h)
	model.logistic0 = NewLogistic(h)
	model.pool = NewDense(h, 1)
	model.loss = NewEuclidean(1)
	return model
}

func (m *Model) Forward(target, x []float64) (float64, float64) {
	hidden := m.logistic0.Forward(m.dense0.Forward(x))
	pred := m.pool.Forward(hidden)
	loss := m.loss.Forward(target, pred)
	return loss, pred[0]
}

func (m *Model) Backward() [4][2][]float64 {
	m.loss.Backward()
	m.pool.Backward(m.loss.Grad()[0][1])
	m.logistic0.Backward(m.pool.Grad()[0][1])
	m.dense0.Backward(m.logistic0.Grad()[0][1])

	var grad [4][2][]float64
	g0 := m.dense0.Grad()
	copy(grad[:2], g0[1:])
	gPool := m.pool.Grad()
	copy(grad[2:], gPool[1:])
	return grad
}

func nnlog() {
	log.Printf("")
}
