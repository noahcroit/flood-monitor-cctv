#!/bin/bash
# Script for setup THOpenSCADA and RTSP-WebRTC for Watergate Project
# set this to 0 if RTSPtoWebRTC is not used
export USE_RTSPTOWEBRTC=0
# set this to 0 if RTSPtoWeb (Full-Version of RTSPtoWebRTC) is not used
export USE_RTSPTOWEB=1
export HOME=/home/pi
export GO=/usr/local/go/bin/go
export GOCACHE=/usr/local/go/bin/cache
export SCADA=$HOME/scada
export WEBRTC=$HOME/RTSPtoWebRTC
echo $HOME
echo $GOCACHE
echo $XDG_CACHE_HOME
timeout=0
while [ 1 ]
do
	# Check SCADA server
	if [ $(docker ps | grep scada/django | wc -l ) -gt 0 ]; 
	then
		echo "scada is running..."
	else
		echo "restart scada"
		cd $SCADA
		docker-compose up -d
	fi

	# Check RTSP-to-WEBRTC server
	if [ $USE_RTSPTOWEBRTC -eq 1 ];
	then
		if [ $(ss -tulpn | grep LISTEN | grep :8083 | wc -l ) -eq 0 ];
		then
			((timeout++))
			if [ $timeout -gt 3 ];
			then
				echo "restart RTSP-to-WebRTC (go version)"
				cd $WEBRTC
				GO111MODULE=on
				$GO run *.go &
			fi
		else
			echo "rtsp-to-webrtc (go version) is running..."
			timeout=0
		fi
	fi

	# Check RTSP-to-WEB server (full-version)
	if [ $USE_RTSPTOWEB -eq 1 ];
	then
		if [ $(docker ps | grep rtsp-to-web | wc -l ) -gt 0 ];
		then	
			echo "rtsp-to-web (docker version) is running..."
		else
			echo "restart rtsp-to-web (docker version)"
			docker restart rtsp-to-web
		fi
	fi

	sleep 10
done

