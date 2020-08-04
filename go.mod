module github.com/paddlepaddle/edl

go 1.13

require (
	github.com/coreos/go-semver v0.3.0 // indirect
	github.com/coreos/go-systemd v0.0.0-20190321100706-95778dfbb74e // indirect
	github.com/coreos/pkg v0.0.0-20180928190104-399ea9e2e55f // indirect
	github.com/dustin/go-humanize v1.0.0 // indirect
	github.com/gogo/protobuf v1.3.1 // indirect
	github.com/golang/protobuf v1.4.1
	github.com/google/uuid v1.1.1
	github.com/gorilla/websocket v1.4.0 // indirect
	github.com/inconshreveable/log15 v0.0.0-20200109203555-b30bc20e4fd1
	github.com/json-iterator/go v1.1.8 // indirect
	github.com/mattn/go-colorable v0.1.6 // indirect
	github.com/namsral/flag v1.7.4-pre
	github.com/spf13/pflag v1.0.5 // indirect
	go.etcd.io/etcd v0.5.0-alpha.5.0.20200401174654-e694b7bb0875
	go.uber.org/zap v1.15.0 // indirect
	golang.org/x/net v0.0.0-20200506145744-7e3656a0809f // indirect
	golang.org/x/sys v0.0.0-20200501145240-bc7a7d42d5c3 // indirect
	golang.org/x/text v0.3.2 // indirect
	google.golang.org/grpc v1.26.0
	gopkg.in/square/go-jose.v2 v2.5.1
	sigs.k8s.io/yaml v1.2.0 // indirect
)

replace (
	github.com/coreos/etcd => github.com/coreos/etcd v3.3.10+incompatible
	github.com/coreos/go-etcd => github.com/coreos/go-etcd v2.0.0+incompatible
	github.com/coreos/go-oidc => github.com/coreos/go-oidc v2.1.0+incompatible
	github.com/coreos/go-semver => github.com/coreos/go-semver v0.3.0
	github.com/coreos/go-systemd => github.com/coreos/go-systemd v0.0.0-20190321100706-95778dfbb74e
	github.com/coreos/pkg => github.com/coreos/pkg v0.0.0-20180108230652-97fdf19511ea // 97fdf19511ea is the SHA for git tag v4
	github.com/grpc-ecosystem/go-grpc-middleware => github.com/grpc-ecosystem/go-grpc-middleware v1.0.1-0.20190118093823-f849b5445de4
	github.com/grpc-ecosystem/go-grpc-prometheus => github.com/grpc-ecosystem/go-grpc-prometheus v1.2.0
	github.com/grpc-ecosystem/grpc-gateway => github.com/grpc-ecosystem/grpc-gateway v1.9.5
)
