package main

import (
	"flag"
	"os"
	"time"

	log "github.com/inconshreveable/log15"

	"k8s.io/api/core/v1"
	extcli "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/leaderelection"
	"k8s.io/client-go/tools/leaderelection/resourcelock"
	"k8s.io/client-go/tools/record"

	paddleclientset "github.com/paddlepaddle/edl/pkg/client/clientset/versioned"
	"github.com/paddlepaddle/edl/pkg/client/clientset/versioned/scheme"
	paddleinformers "github.com/paddlepaddle/edl/pkg/client/informers/externalversions"
	paddlecontroller "github.com/paddlepaddle/edl/pkg/controller"
	"github.com/paddlepaddle/edl/pkg/signals"
)

var (
	leaseDuration = 15 * time.Second
	renewDuration = 5 * time.Second
	retryPeriod   = 3 * time.Second
)

func main() {
	masterURL := flag.String("master", "", "Address of a kube master.")
	kubeConfig := flag.String("kubeconfig", "", "Path to a kube config. Only required if out-of-cluster.")
	flag.Parse()

	stopCh := signals.SetupSignalHandler()

	cfg, err := clientcmd.BuildConfigFromFlags(*masterURL, *kubeConfig)
	if err != nil {
		log.Error("Error building kubeconfig:", err.Error())
		return
	}

	kubeClient, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		log.Error("Error building kubernetes clientset:", err.Error())
		return
	}

	extapiClient, err := extcli.NewForConfig(cfg)
	if err != nil {
		log.Error("Error building kubernetes extension api clientset:", err.Error())
		return
	}

	paddleClient, err := paddleclientset.NewForConfig(cfg)
	if err != nil {
		log.Error("Error building paddle clientset:", err.Error())
		return
	}

	paddleInformer := paddleinformers.NewSharedInformerFactory(paddleClient, time.Second*10)

	controller := paddlecontroller.New(kubeClient, extapiClient, paddleClient, paddleInformer)

	hostname, err := os.Hostname()
	if err != nil {
		log.Error("Error checking hostname:", err.Error())
		return
	}

	go paddleInformer.Start(stopCh)

	run := func(stop <-chan struct{}) {
		log.Info("I won the leader election")
		if controller.Run(1, stopCh); err != nil {
			log.Error("Error running paddle trainingjob controller:", err.Error())
			return
		}
	}

	stop := func() {
		log.Error("I lost the leader election")
		return
	}

	leaderElectionClient, err := kubernetes.NewForConfig(rest.AddUserAgent(cfg, "leader-election"))
	if err != nil {
		log.Error("Error building leader election clientset:", err.Error())
		return
	}

	// Prepare event clients.
	eventBroadcaster := record.NewBroadcaster()
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "trainingjob-controller"})

	lock := &resourcelock.EndpointsLock{
		EndpointsMeta: metav1.ObjectMeta{
			Namespace: "kube-system",
			Name:      "trainingjob-controller",
		},
		Client: leaderElectionClient.CoreV1(),
		LockConfig: resourcelock.ResourceLockConfig{
			Identity:      hostname,
			EventRecorder: recorder,
		},
	}

	leaderelection.RunOrDie(leaderelection.LeaderElectionConfig{
		Lock:          lock,
		LeaseDuration: leaseDuration,
		RenewDeadline: renewDuration,
		RetryPeriod:   retryPeriod,
		Callbacks: leaderelection.LeaderCallbacks{
			OnStartedLeading: run,
			OnStoppedLeading: stop,
		},
	})
}
