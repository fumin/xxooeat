package model

import (
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/fumin/xxooeat/nn"
)

func TestDropoutLearn(t *testing.T) {
	t.Parallel()
	x := make([]float64, 4)
	model := NewDropout([]int{len(x), 32, 32})
	grads := model.Backward()
	optimizer := nn.NewAdam(grads[:], 1e-3, 0.9, 0.999, 1e-7)

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
	for i := 0; i < 40000; i++ {
		target, x := newDatum()
		model.Forward(target, x)
		gradients := model.Backward()
		optimizer.Update(gradients)
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
		// log.Printf("x: %+v, target: %d, pred: %d", x, int(target[0]), int(pred))
	}
	accuracy := sum(trials) / float64(len(trials))
	if accuracy < 0.95 {
		t.Fatalf("%f", accuracy)
	}
}

func TestDropoutSave(t *testing.T) {
	t.Parallel()
	dir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("%+v", err)
	}
	savePath := filepath.Join(dir, "mlp")

	model := NewDropout([]int{2, 3, 3})
	if err := model.Save(savePath); err != nil {
		t.Fatalf("%+v", err)
	}

	b, err := os.ReadFile(savePath)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	loaded, err := LoadDropout(b)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	if !reflect.DeepEqual(loaded, model) {
		t.Fatalf("%+v %+v", loaded, model)
	}
	// log.Printf("%+v %+v", loaded.dense0, loaded.pool)
}

func dropoutlog() {
	log.Printf("")
}
