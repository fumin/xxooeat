package main

import (
	"flag"
	"log"

	"github.com/fumin/xxooeat"
)

func main() {
	flag.Parse()
	log.SetFlags(log.LstdFlags | log.Lmicroseconds | log.Lshortfile)

	model := xxooeat.NewDense(81, 128)
	log.Printf("%+v", model)
}
