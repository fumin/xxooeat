package main

import (
	"encoding/json"
	"flag"
	//	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	//	"strings"

	"github.com/fumin/xxooeat"
	"github.com/fumin/xxooeat/model"
	"github.com/fumin/xxooeat/nn"
)

var (
	cfgStr = flag.String("config", `{
		"Dir": "/Users/mac/fumin/tmp/xxooeat/mc128_128",
		"Seed": 0,

		"BoardSize": 3,
		"MaxPieceSize": 3,
		"NumPiecesPerSize": 2,
		"CanMove": true,

		"Epsilon": 1e-1,
		"HiddenSize": [128, 128],
		"LearningRate": 1e-3
	}`, "experiment configuration")
)

func main() {
	flag.Parse()
	log.SetFlags(log.LstdFlags | log.Lmicroseconds | log.Lshortfile)

	cfg := Config{}
	if err := json.Unmarshal([]byte(*cfgStr), &cfg); err != nil {
		log.Fatalf("%+v", err)
	}
	log.Printf("CONFIG %s", *cfgStr)
	if err := os.MkdirAll(cfg.Dir, 0755); err != nil {
		log.Fatalf("%+v", err)
	}
	rand.Seed(cfg.Seed)

	state := xxooeat.NewState(cfg.BoardSize, cfg.MaxPieceSize, cfg.NumPiecesPerSize)
	valueNet := model.NewMLP(append([]int{len(state.S)}, cfg.HiddenSize...))
	savePath := filepath.Join(cfg.Dir, "model")
	net, err := func(savePath string) (*model.MLP, error) {
		b, err := os.ReadFile(savePath)
		if err != nil {
			return nil, errors.Wrap(err, "")
		}
		valueNet, err := model.LoadMLP(b)
		if err != nil {
			return nil, errors.Wrap(err, "")
		}
		return valueNet, nil
	}(savePath)
	if err != nil {
		log.Printf("new model")
	} else {
		valueNet = net
		log.Printf("successfully loaded from \"%s\"", savePath)
	}
	grads := valueNet.Backward()
	optimizer := nn.NewAdam(grads[:], 1e-3, 0.9, 0.999, 1e-7)
	experienceReplay := NewExperienceReplay()

	actChan := make(chan Datum, 0)
	learnChan := make(chan struct{}, 0)
	go func() {
		for {
			play(cfg, valueNet, actChan, learnChan)
		}
	}()

	//f, err := os.Create("/Users/mac/fumin/tmp/xxooeat/data")
	//if err != nil {
	//	log.Fatalf("%+v", err)
	//}

	go func() {
		<-learnChan
	}()
	for step := 0; step < 99999999; step++ {
		learnChan <- struct{}{}
		d := <-actChan

		//fs := []string{}
		//for _, f := range d.State.S {
		//	fs = append(fs, fmt.Sprintf("%d", int(f)))
		//}
		//fs = append(fs, fmt.Sprintf("%d",int(d.Reward)))
		//if _, err := f.Write([]byte(strings.Join(fs, ",")+"\n")); err != nil {
		//	log.Fatalf("%+v", err)
		//}

		experienceReplay.Add(d)
		d = experienceReplay.Get()
		if step < experienceReplay.Size() {
			continue
		}

		var loss float64
		loss, _ = valueNet.Forward([]float64{d.Reward}, d.State.S)
		gradients := valueNet.Backward()
		optimizer.Update(gradients[:])

		if step%1000 == 0 {
			log.Printf("reward: %f, player: %d, state: %s", d.Reward, d.State.Player(), d.State)
			if err := valueNet.Save(savePath); err != nil {
				log.Fatalf("%+v", err)
			}
			vals := struct {
				Step int
				Loss float64
			}{
				Step: step,
				Loss: loss,
			}
			b, err := json.Marshal(vals)
			if err != nil {
				log.Fatalf("%+v", err)
			}
			log.Printf("LOG %s", b)
		}
	}
}

type Datum struct {
	State      *xxooeat.State
	Reward     float64
	IsTerminal bool
}

type ExperienceReplay struct {
	data   []Datum
	cursor int
}

func NewExperienceReplay() *ExperienceReplay {
	er := &ExperienceReplay{}
	er.data = make([]Datum, 1024*64)
	return er
}

func (er *ExperienceReplay) Add(d Datum) {
	if er.cursor >= len(er.data) {
		er.cursor = 0
	}
	er.data[er.cursor] = d
	er.cursor++
}

func (er *ExperienceReplay) Get() Datum {
	idx := rand.Intn(len(er.data))
	return er.data[idx]
}

func (er *ExperienceReplay) Size() int {
	return len(er.data)
}

func play(cfg Config, valueNet *model.MLP, actChan chan Datum, learnChan chan struct{}) {
	history := []*xxooeat.State{xxooeat.NewState(cfg.BoardSize, cfg.MaxPieceSize, cfg.NumPiecesPerSize)}
	history[0].CanMove = cfg.CanMove
	var noMove = false
	var win [2]bool
	for {
		nextState := act(cfg.Epsilon, valueNet, history[len(history)-1])
		if nextState == nil {
			noMove = true
			break
		}

		history = append(history, nextState)
		win = history[len(history)-1].Win()
		if win[0] || win[1] {
			break
		}
	}
	for i := 0; i < len(history)-1; i++ {
		var isTerminal bool
		if i == len(history)-2 || i == len(history)-3 {
			isTerminal = true
		}

		var reward float64
		switch {
		case noMove:
			reward = 0
		case win[0] && win[1]:
			reward = 0
		case win[history[i].Player()]:
			reward = 1
		default:
			reward = -1
		}
		actChan <- Datum{State: history[i+1], Reward: reward, IsTerminal: isTerminal}
		<-learnChan
	}
}

func act(epsilon float64, valueNet *model.MLP, state *xxooeat.State) *xxooeat.State {
	nexts := state.Next()
	if len(nexts) == 0 {
		return nil
	}

	if rand.Float64() < epsilon {
		return nexts[rand.Intn(len(nexts))].State
	}

	var maxIdx int = -1
	var max float64 = -math.MaxFloat64
	for i, a := range nexts {
		_, v := valueNet.Forward([]float64{-1}, a.State.S)
		if v > max {
			maxIdx = i
			max = v
		}
	}

	return nexts[maxIdx].State
}

type Config struct {
	Dir  string
	Seed int64

	BoardSize        int
	MaxPieceSize     int
	NumPiecesPerSize int
	CanMove          bool

	Epsilon      float64
	HiddenSize   []int
	LearningRate float64
}
