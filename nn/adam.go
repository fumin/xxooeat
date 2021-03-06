package nn

import (
	"math"
)

type Adam struct {
	learningRate float64
	beta1        float64
	beta2        float64
	epsilon      float64
	m            [][]float64
	v            [][]float64
}

func NewAdam(grad [][2][]float64, learningRate, beta1, beta2, epsilon float64) *Adam {
	optmz := &Adam{}
	optmz.learningRate = learningRate
	optmz.beta1 = beta1
	optmz.beta2 = beta2
	optmz.epsilon = epsilon

	optmz.m = make([][]float64, len(grad))
	optmz.v = make([][]float64, len(grad))
	for i, g := range grad {
		optmz.m[i] = make([]float64, len(g[1]))
		optmz.v[i] = make([]float64, len(g[1]))
	}

	return optmz
}

func (optmz *Adam) Update(gradient [][2][]float64) {
	for i, g := range gradient {
		weights := g[0]
		grads := g[1]
		for j := 0; j < len(weights); j++ {
			optmz.m[i][j] = optmz.beta1*optmz.m[i][j] + (1-optmz.beta1)*grads[j]
			optmz.v[i][j] = optmz.beta2*optmz.v[i][j] + (1-optmz.beta2)*grads[j]*grads[j]
			weights[j] += -optmz.learningRate * optmz.m[i][j] / (math.Sqrt(optmz.v[i][j]) + optmz.epsilon)
		}
	}
}
