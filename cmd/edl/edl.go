package main

import (
	"flag"
	"os"
	"time"

	log "github.com/inconshreveable/log15"
	"github.com/wangkuiyi/candy"

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
	autoClean := flag.Bool("autoclean", false, "Auto clean pods after terminating job, default false")
	maxLoadDesired := flag.Float64("max_load_desired", 0.97, `Keep the cluster max resource usage around
		this value, jobs will scale down if total request is over this level.`)
	flag.Parse()

	stopCh := signals.SetupSignalHandler()

	cfg, err := clientcmd.BuildConfigFromFlags(*masterURL, *kubeConfig)
	candy.Must(err)

	kubeClient, err := kubernetes.NewForConfig(cfg)
	candy.Must(err)

	extapiClient, err := extcli.NewForConfig(cfg)
	candy.Must(err)

	paddleClient, err := paddleclientset.NewForConfig(cfg)
	candy.Must(err)

	hostname, err := os.Hostname()
	candy.Must(err)

	run := func(stop <-chan struct{}) {
		log.Info("I won the leader election", "hostname", hostname)
		paddleInformer := paddleinformers.NewSharedInformerFactory(paddleClient, time.Second*10)
		controller := paddlecontroller.New(kubeClient, extapiClient, paddleClient, paddleInformer, *autoClean)
		go paddleInformer.Start(stopCh)

		if controller.Run(1, *maxLoadDesired, stopCh); err != nil {
			log.Error("Error running paddle trainingjob controller", "error", err.Error())
			return
		}
	}

	stop := func() {
		log.Error("I lost the leader election", "hostname", hostname)
		return
	}

	leaderElectionClient, err := kubernetes.NewForConfig(rest.AddUserAgent(cfg, "leader-election"))
	if err != nil {
		log.Error("Error building leader election clientset", "error", err.Error())
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
