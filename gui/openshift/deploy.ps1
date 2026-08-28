param(
  [string]$Project = "",
  [string]$CreateProject = "",
  [ValidateSet("openshift_mock", "openshift_job")]
  [string]$Runtime = $(if ($env:CONDITAR_RUNTIME) { $env:CONDITAR_RUNTIME } else { "openshift_mock" }),
  [string]$RuntimeImage = $(if ($env:CONDITAR_DOCKER_IMAGE) { $env:CONDITAR_DOCKER_IMAGE } else { "osuninglab/conditar-dev:2026-07-10" }),
  [switch]$Submit,
  [switch]$Cpu,
  [string]$Storage = $(if ($env:CONDITAR_OPENSHIFT_STORAGE) { $env:CONDITAR_OPENSHIFT_STORAGE } else { "10Gi" }),
  [string]$RouteHost = $(if ($env:CONDITAR_OPENSHIFT_ROUTE_HOST) { $env:CONDITAR_OPENSHIFT_ROUTE_HOST } else { "" }),
  [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

function Run-Oc {
  & oc @args
  if ($LASTEXITCODE -ne 0) {
    throw "oc command failed: oc $($args -join ' ')"
  }
}

function Get-OcText {
  $output = & oc @args
  if ($LASTEXITCODE -ne 0) {
    return ""
  }
  return ($output -join "`n").Trim()
}

$guiRoot = Split-Path -Parent $PSScriptRoot
Set-Location $guiRoot

if (-not (Get-Command oc -ErrorAction SilentlyContinue)) {
  throw "oc.exe was not found. Install the OpenShift CLI, then run oc login first."
}

Run-Oc whoami | Out-Null

if ($CreateProject) {
  & oc new-project $CreateProject | Out-Null
  if ($LASTEXITCODE -ne 0) {
    Run-Oc project $CreateProject | Out-Null
  }
} elseif ($Project) {
  Run-Oc project $Project | Out-Null
}

$projectName = Get-OcText project -q
Write-Host "Using OpenShift project: $projectName"

$openshiftSubmit = if ($Submit -or ($env:CONDITAR_OPENSHIFT_SUBMIT -match "^(1|true|yes|on)$")) { "true" } else { "false" }
$openshiftDevice = if ($env:CONDITAR_OPENSHIFT_DEVICE) { $env:CONDITAR_OPENSHIFT_DEVICE } else { "cuda:0" }
$openshiftGpuCount = if ($env:CONDITAR_OPENSHIFT_GPU_COUNT) { $env:CONDITAR_OPENSHIFT_GPU_COUNT } else { "1" }
$openshiftCpuRequest = if ($env:CONDITAR_OPENSHIFT_CPU_REQUEST) { $env:CONDITAR_OPENSHIFT_CPU_REQUEST } else { "2" }
$openshiftMemoryRequest = if ($env:CONDITAR_OPENSHIFT_MEMORY_REQUEST) { $env:CONDITAR_OPENSHIFT_MEMORY_REQUEST } else { "16Gi" }
$openshiftMemoryLimit = if ($env:CONDITAR_OPENSHIFT_MEMORY_LIMIT) { $env:CONDITAR_OPENSHIFT_MEMORY_LIMIT } else { "32Gi" }

if ($Cpu) {
  $openshiftDevice = "cpu"
  $openshiftGpuCount = "0"
  $openshiftCpuRequest = if ($env:CONDITAR_OPENSHIFT_CPU_REQUEST) { $env:CONDITAR_OPENSHIFT_CPU_REQUEST } else { "500m" }
  $openshiftMemoryRequest = if ($env:CONDITAR_OPENSHIFT_MEMORY_REQUEST) { $env:CONDITAR_OPENSHIFT_MEMORY_REQUEST } else { "4Gi" }
  $openshiftMemoryLimit = if ($env:CONDITAR_OPENSHIFT_MEMORY_LIMIT) { $env:CONDITAR_OPENSHIFT_MEMORY_LIMIT } else { "8Gi" }
}

$previousImageRef = Get-OcText get deployment/conditar-gui -o "jsonpath={.spec.template.spec.containers[?(@.name==`"gui`")].image}"
$existingImageRef = Get-OcText get istag conditar-gui:latest -o "jsonpath={.image.dockerImageReference}"
$currentImageRef = if ($previousImageRef) { $previousImageRef } else { $existingImageRef }

$tmpRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("conditar-openshift-" + [System.Guid]::NewGuid().ToString("N"))
$tmpOpenShift = Join-Path $tmpRoot "openshift"
New-Item -ItemType Directory -Path $tmpRoot | Out-Null

try {
  Copy-Item -Recurse -Path (Join-Path $guiRoot "openshift") -Destination $tmpOpenShift

  $configPath = Join-Path $tmpOpenShift "configmap.yaml"
  $config = Get-Content $configPath -Raw
  $replacements = @{
    "CONDITAR_RUNTIME" = $Runtime
    "CONDITAR_DOCKER_IMAGE" = $RuntimeImage
    "CONDITAR_OPENSHIFT_SUBMIT" = $openshiftSubmit
    "CONDITAR_OPENSHIFT_DEVICE" = $openshiftDevice
    "CONDITAR_OPENSHIFT_GPU_COUNT" = $openshiftGpuCount
    "CONDITAR_OPENSHIFT_CPU_REQUEST" = $openshiftCpuRequest
    "CONDITAR_OPENSHIFT_MEMORY_REQUEST" = $openshiftMemoryRequest
    "CONDITAR_OPENSHIFT_MEMORY_LIMIT" = $openshiftMemoryLimit
  }
  foreach ($key in $replacements.Keys) {
    $escaped = [regex]::Escape($key)
    $value = $replacements[$key]
    $config = $config -replace "$escaped`: `"[^`"]*`"", "$key`: `"$value`""
  }
  Set-Content -Path $configPath -Value $config -NoNewline

  $pvcPath = Join-Path $tmpOpenShift "pvc.yaml"
  $pvc = Get-Content $pvcPath -Raw
  $pvc = $pvc -replace "storage: .*", "storage: $Storage"
  Set-Content -Path $pvcPath -Value $pvc -NoNewline

  if ($RouteHost) {
    $routePath = Join-Path $tmpOpenShift "route.yaml"
    $route = Get-Content $routePath -Raw
    if ($route -match "`n  host: ") {
      $route = $route -replace "`n  host: .*", "`n  host: $RouteHost"
    } else {
      $route = $route -replace "`nspec:`n", "`nspec:`n  host: $RouteHost`n"
    }
    Set-Content -Path $routePath -Value $route -NoNewline
  }

  if ($currentImageRef) {
    $deploymentPath = Join-Path $tmpOpenShift "deployment.yaml"
    $deployment = Get-Content $deploymentPath -Raw
    $deployment = $deployment -replace "image: conditar-gui:[^\s]+", "image: $currentImageRef"
    Set-Content -Path $deploymentPath -Value $deployment -NoNewline
  }

  Run-Oc apply -k $tmpOpenShift

  if (-not $SkipBuild) {
    Write-Host "Starting OpenShift binary build from $guiRoot"
    & oc start-build conditar-gui --from-dir=. --follow --wait
    if ($LASTEXITCODE -ne 0) {
      if ($previousImageRef) {
        Write-Host "Build failed; restoring previous GUI image."
        & oc set image deployment/conditar-gui "gui=$previousImageRef" | Out-Null
      }
      exit 1
    }
  } else {
    Write-Host "Skipping build because -SkipBuild was provided."
  }

  $imageRef = Get-OcText get istag conditar-gui:latest -o "jsonpath={.image.dockerImageReference}"
  if ($imageRef) {
    Run-Oc set image deployment/conditar-gui "gui=$imageRef" | Out-Null
  }

  Run-Oc rollout status deployment/conditar-gui

  $routeUrl = Get-OcText get route conditar-gui -o "jsonpath={.spec.host}"
  if ($routeUrl) {
    Write-Host "conDitar GUI: https://$routeUrl"
  } else {
    Write-Host "Deployment is ready. No Route host was reported; inspect with: oc get route conditar-gui"
  }
} finally {
  Remove-Item -Recurse -Force $tmpRoot -ErrorAction SilentlyContinue
}
