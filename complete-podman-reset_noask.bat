@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

echo [BEGIN] Podman environment reset

echo [BEGIN] Stop all containers
podman stop --all
echo [END] Stop all containers

echo [BEGIN] Remove all containers
podman rm --all --force
echo [END] Remove all containers

echo [BEGIN] Remove existing podman machine
podman machine stop
podman machine rm --force
echo [END] Remove existing podman machine

echo [BEGIN] Initialize new podman machine
podman machine init
echo [END] Initialize new podman machine

echo [BEGIN] Start new podman machine
podman machine start
echo [END] Start new podman machine

echo [BEGIN] Cleanup unused volumes
podman volume prune --force
echo [END] Cleanup unused volumes

echo [END] Podman environment reset
exit /b 0
