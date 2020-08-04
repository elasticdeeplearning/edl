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
	"fmt"
	"net"
	"os"
	"os/signal"
	"strings"
	"time"

	log "github.com/inconshreveable/log15"
	"github.com/namsral/flag"
	master "github.com/paddlepaddle/edl/pkg/master"
	pb "github.com/paddlepaddle/edl/pkg/masterpb"
	utils "github.com/paddlepaddle/edl/pkg/utils"
	grpc "google.golang.org/grpc"
)

type args struct {
	port           uint
	ttlSec         int
	jobID          string
	etcd_endpoints []string
	taskTimeoutDur time.Duration
	taskTimeoutMax int
	logLevel       string
	local_ip       string
	listen_port    int
	endpoint       string
}

func parseArgs() *args {
	port := flag.Uint("port", 0, "port of the master server.")
	ttlSec := flag.Int("ttl", 10, "etcd lease TTL in seconds.")
	etcd_endpoints := flag.String("endpoints", "http://127.0.0.1:2379", "comma separated etcd endpoints. If empty, fault tolerance will not be enabled.")
	jobID := flag.String("jobID", "", "jobID of this master")
	taskTimeoutDur := flag.Duration("task-timout-dur", 20*time.Minute, "task timout duration.")
	taskTimeoutMax := flag.Int("task-timeout-max", 3, "max timtout count for each task before it being declared failed task.")
	logLevel := flag.String("log-level", "info",
		"log level, possible values: debug, info, warn, error, crit")
	flag.Parse()

	if *jobID == "" {
		panic("jobID must set")
	}

	if *etcd_endpoints == "" {
		log.Crit("-endpoints not set!.")
		panic("")
	}

	eps := strings.Split(*etcd_endpoints, ",")
	ip, err := utils.GetExternalIP()
	if err != nil {
		log.Crit("get external ip error", log.Ctx{"error": err})
		panic(err)
	}

	a := &args{}
	a.port = *port
	a.ttlSec = *ttlSec
	a.etcd_endpoints = eps
	a.local_ip = ip
	a.jobID = *jobID
	a.taskTimeoutMax = *taskTimeoutMax
	a.taskTimeoutDur = *taskTimeoutDur
	a.logLevel = *logLevel
	a.endpoint = fmt.Sprintf("%s:%d", a.local_ip, a.port)

	return a
}

func main() {
	a := parseArgs()

	lvl, err := log.LvlFromString(a.logLevel)
	if err != nil {
		panic(err)
	}

	log.Root().SetHandler(
		log.LvlFilterHandler(lvl, log.CallerStackHandler("%+v", log.StderrHandler)),
	)

	c := make(chan os.Signal)
	signal.Notify(c, os.Interrupt)

	store := master.Election(a.jobID, a.etcd_endpoints, a.ttlSec)
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

	s, err := master.NewService(a.jobID, store, a.taskTimeoutDur, a.taskTimeoutMax)
	if err != nil {
		log.Crit("error creating new service.", log.Ctx{"error": err})
		panic(err)
	}

	grpcServer := grpc.NewServer()
	pb.RegisterMasterServer(grpcServer, s)

	listener, err := net.Listen("tcp", fmt.Sprintf(":%v", a.port))
	if err != nil {
		log.Crit("could not listen to 0.0.0.0:%v %v", a.port, err)
		panic(err)
	}
	a.listen_port = listener.Addr().(*net.TCPAddr).Port
	a.endpoint = fmt.Sprintf("%s:%d", a.local_ip, a.listen_port)

	s.Register(a.endpoint)

	go func() {
		log.Info("Server starting...")
		err = grpcServer.Serve(listener)
		if err != nil {
			log.Crit("error serving", log.Ctx{"error": err})
			panic(err)
		}
	}()

	<-c
}
