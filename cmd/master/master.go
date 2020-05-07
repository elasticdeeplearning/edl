// Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"os"
	"os/signal"
	"time"

	log "github.com/inconshreveable/log15"
	"github.com/namsral/flag"
	master "github.com/paddlepaddle/edl/pkg/master"
)

func main() {
	port := flag.Int("port", 8080, "port of the master server.")
	ttlSec := flag.Int("ttl", 10, "etcd lease TTL in seconds.")
	endpoints := flag.String("endpoints", "http://127.0.0.1:2379", "comma separated etcd endpoints. If empty, fault tolerance will not be enabled.")
	taskTimeoutDur := flag.Duration("task-timout-dur", 20*time.Minute, "task timout duration.")
	taskTimeoutMax := flag.Int("task-timeout-max", 3, "max timtout count for each task before it being declared failed task.")
	//chunkPerTask := flag.Int("chunk-per-task", 10, "chunk per task.")
	logLevel := flag.String("log-level", "info",
		"log level, possible values: debug, info, warn, error, crit")
	flag.Parse()

	lvl, err := log.LvlFromString(*logLevel)
	if err != nil {
		panic(err)
	}

	log.Root().SetHandler(
		log.LvlFilterHandler(lvl, log.CallerStackHandler("%+v", log.StderrHandler)),
	)

	if *endpoints == "" {
		log.Fatal("-endpoints not set!.")
	}

	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt)

	eps := strings.Split(*endpoints, ",")
	ip, err := networkhelper.GetExternalIP()
	if err != nil {
		log.Crit("get external ip error", log.Ctx{"error": err})
		panic(err)
	}

	addr := fmt.Sprintf("%s:%d", ip, *port)
	store, err = master.NewEtcdClient(eps, addr, master.DefaultLockPath, master.DefaultAddrPath, master.DefaultStatePath, *ttlSec)
	if err != nil {
		log.Crit("error creating etcd client.", log.Ctx{"error": err})
		panic(err)
	}

	shutdown := func() {
		log.Info("shutting down gracefully")
		err := store.Shutdown()
		if err != nil {
			log.Error("shutdown error", log.Ctx{"error": err})
		}
	}

	// Guaranteed to run even panic happens.
	defer shutdown()

	s, err := master.NewService(store, *taskTimeoutDur, *taskTimeoutMax)
	if err != nil {
		log.Crit("error creating new service.", log.Ctx{"error": err})
		panic(err)
	}

	grpcServer := grpc.NewServer()
	countries.RegisterCountryServer(grpcServer, s)
	listen, err := net.Listen("tcp", fmt.Sprintf(":%v", *port))
	if err != nil {
		log.Fatalf("could not listen to 0.0.0.0:3000 %v", err)
	}

	go func() {
		log.Println("Server starting...")
		err = grpcServer.Serve(listen)
		if err != nil {
			log.Crit("error serving", log.Ctx{"error": err})
			panic(err)
		}
	}()

	<-c
}
