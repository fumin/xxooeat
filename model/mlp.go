package model

import (
	"encoding/json"
	"os"

	"github.com/fumin/xxooeat/nn"
	"github.com/pkg/errors"
)

type MLP struct {
	sizes      []int
	dense      []*nn.Dense
	activation []*nn.Tanh
	pool       *nn.Dense
	loss       *nn.Euclidean
}

func NewMLP(sizes []int) *MLP {
	m := &MLP{}
	m.sizes = make([]int, len(sizes))
	copy(m.sizes, sizes)
	for i := 1; i < len(sizes); i++ {
		m.dense = append(m.dense, nn.NewDense(sizes[i-1], sizes[i]))
		m.activation = append(m.activation, nn.NewTanh(sizes[i]))
	}
	m.pool = nn.NewDense(sizes[len(sizes)-1], 1)
	m.loss = nn.NewEuclidean(1)
	return m
}

type MLPData struct {
	Name   string
	Sizes  []int
	DenseW [][]float64
	DenseB [][]float64
	PoolW  []float64
	PoolB  []float64
}

func LoadMLP(b []byte) (*MLP, error) {
	data := MLPData{}
	if err := json.Unmarshal(b, &data); err != nil {
		return nil, errors.Wrap(err, "")
	}
	m := NewMLP(data.Sizes)
	for i, w := range data.DenseW {
		copy(m.dense[i].W, w)
		copy(m.dense[i].B, data.DenseB[i])
	}
	copy(m.pool.W, data.PoolW)
	copy(m.pool.B, data.PoolB)
	return m, nil
}

func (m *MLP) Forward(target, x []float64) (float64, float64) {
	hidden := make([]float64, len(x))
	for i, xi := range x {
		hidden[i] = xi*2 - 1
	}

	for i, dense := range m.dense {
		hidden = m.activation[i].Forward(dense.Forward(hidden))
	}
	pred := m.pool.Forward(hidden)
	loss := m.loss.Forward(target, pred)
	return loss, pred[0]
}

func (m *MLP) Backward() [][2][]float64 {
	m.loss.Backward()
	m.pool.Backward(m.loss.Grad()[0][1])
	grad := m.pool.Grad()[0][1]
	for i := len(m.dense) - 1; i >= 0; i-- {
		m.activation[i].Backward(grad)
		m.dense[i].Backward(m.activation[i].Grad()[0][1])
		grad = m.dense[i].Grad()[0][1]
	}

	wg := make([][2][]float64, 0)
	for _, dense := range m.dense {
		wgi := dense.Grad()
		wg = append(wg, wgi[1])
		wg = append(wg, wgi[2])
	}
	wgPool := m.pool.Grad()
	wg = append(wg, wgPool[1])
	wg = append(wg, wgPool[2])
	return wg
}

func (m *MLP) Save(savePath string) error {
	data := MLPData{}
	data.Name = "MLP"
	data.Sizes = make([]int, len(m.sizes))
	copy(data.Sizes, m.sizes)
	data.DenseW = make([][]float64, len(m.dense))
	data.DenseB = make([][]float64, len(m.dense))
	for i, dense := range m.dense {
		data.DenseW[i] = dense.W
		data.DenseB[i] = dense.B
	}
	data.PoolW = m.pool.W
	data.PoolB = m.pool.B

	b, err := json.Marshal(data)
	if err != nil {
		return errors.Wrap(err, "")
	}
	if err := os.WriteFile(savePath, b, 0644); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}
