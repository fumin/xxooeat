package model

import (
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestMLPGrad(t *testing.T) {
	t.Parallel()
	mlp := NewMLP([]int{2, 3, 3})
	x := []float64{0, 1}
	target := []float64{0}
	if b(x[0]) || b(x[1]) {
		target[0] = 1
	}

	for i := 5; i >= 0; i-- {
		mlp.Forward(target, x)
		w := mlp.Backward()[i][0]
		oriW := make([]float64, len(w))
		copy(oriW, w)
		for j := 0; j < len(w); j++ {
			y, _ := mlp.Forward(target, x)

			w[j] += 1e-6
			yy, _ := mlp.Forward(target, x)

			g := (yy - y) / (w[j] - oriW[j])

			copy(w, oriW)
			mlp.Forward(target, x)
			grad := mlp.Backward()
			if math.Abs(g-grad[i][1][j]) > 1e-5 {
				t.Fatalf("weightID: %d, j: %d, trueG: %f, computedG: %+v", i, j, g, grad[i][1][j])
			}
		}
	}
}

func TestMLPLearn(t *testing.T) {
	t.Parallel()
	x := make([]float64, 4)
	mlp := NewMLP([]int{len(x), 3, 3})

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
		mlp.Forward(target, x)
		gradients := mlp.Backward()
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
		_, predF := mlp.Forward(target, x)
		var pred float64 = 0
		if predF > 0.5 {
			pred = 1
		}

		if pred == target[0] {
			trials[i] = 1
		}
		// log.Printf("x: %+v, target: %d, pred: %d", x, int(target[0]), int(pred))
	}
	accuracy := sum(trials) / float64(len(trials))
	if accuracy != 1 {
		t.Fatalf("%f", accuracy)
	}
}

func TestMLPSave(t *testing.T) {
	t.Parallel()
	dir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("%+v", err)
	}
	savePath := filepath.Join(dir, "mlp")

	mlp := NewMLP([]int{2, 3, 3})
	if err := mlp.Save(savePath); err != nil {
		t.Fatalf("%+v", err)
	}

	b, err := os.ReadFile(savePath)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	loaded, err := LoadMLP(b)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	if !reflect.DeepEqual(loaded, mlp) {
		t.Fatalf("%+v %+v", loaded, mlp)
	}
	// log.Printf("%+v %+v", loaded.dense0, loaded.pool)
}

func b(x float64) bool {
	if int(x) == 0 {
		return false
	}
	return true
}

func sum(x []float64) float64 {
	var y float64 = 0
	for i := 0; i < len(x); i++ {
		y += x[i]
	}
	return y
}

func mlplog() {
	log.Printf("")
}
