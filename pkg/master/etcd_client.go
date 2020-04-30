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

package edl

import (
	"context"
	"strings"
	"time"

	log "github.com/inconshreveable/log15"
	"github.com/paddlepaddle/edl/pkg/utils/networkhelper"
	"go.etcd.io/etcd/clientv3"
	"go.etcd.io/etcd/clientv3/concurrency"
)

const (
	retryTimeout = 5 * time.Second
)

// EtcdClient is the etcd client that the pserver uses for fault
// tolerance, service registry and coordination.
type EtcdClient struct {
	numPservers int
	endpoints   string
	client      *clientv3.Client
	sess        *concurrency.Session
	dialTimeout time.Duration
	ttlSec      int
	// FIXME: ensure GetExternalIP gets the correct ip for trainers to connect.
	externalIP string
	// desired number of pservers in the job.
	// assume desired will not change during one training job.
	desired int
}

// NewEtcdClient creates an EtcdClient
func NewEtcdClient(endpoints string, numPservers int, dialtimeout time.Duration, ttlSec int) *EtcdClient {
	return &EtcdClient{
		dialTimeout: dialtimeout,
		ttlSec:      ttlSec,
		numPservers: numPservers,
		endpoints:   endpoints,
	}
}

// Register returns the index of the current pserver.
func (e *EtcdClient) Register(port int) error {
	var err error
	e.externalIP, err = networkhelper.GetExternalIP()
	if err != nil {
		return err
	}

	// initialize connection to etcd.
	ep := strings.Split(e.endpoints, ",")
	for {
		cli, err := clientv3.New(clientv3.Config{
			Endpoints:   ep,
			DialTimeout: e.dialTimeout,
		})
		if err != nil {
			log.Error("connect to etcd error", log.Ctx{"error": err})
			time.Sleep(retryTimeout)
			continue
		}
		e.client = cli
		sess, err := concurrency.NewSession(cli, concurrency.WithTTL(e.ttlSec))
		if err != nil {
			log.Error("create etcd session error", log.Ctx{"error": err})
			time.Sleep(retryTimeout)
			continue
		}
		e.sess = sess
		log.Debug("connected to etcd", log.Ctx{"endpoint": e.endpoints})
		break
	}

	return nil
}

// GetKey gets the value by the specified key
func (e *EtcdClient) GetKey(key string, timeout time.Duration) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	resp, err := e.client.Get(ctx, key)
	cancel()
	if err != nil {
		return []byte{}, err
	}

	kvs := resp.Kvs
	if len(kvs) == 0 {
		return []byte{}, nil
	}
	v := kvs[0].Value
	return v, nil
}

// PutKey put into etcd with value by key specified
func (e *EtcdClient) PutKey(key string, value []byte, timeout time.Duration, withLease bool) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	var err error
	if withLease {
		_, err = e.client.Put(ctx, key, string(value), clientv3.WithLease(e.sess.Lease()))
	} else {
		_, err = e.client.Put(ctx, key, string(value))
	}
	cancel()
	return err
}

// Shutdown shuts down the etcd client gracefully.
func (e *EtcdClient) Shutdown() error {
	var err error
	if e.sess != nil {
		err = e.sess.Close()
	}

	if e.client != nil {
		newErr := e.client.Close()
		if newErr != nil {
			if err != nil {
				log.Error("shutdown error", log.Ctx{"error": newErr})
			} else {
				err = newErr
			}
		}
	}
	return err
}
