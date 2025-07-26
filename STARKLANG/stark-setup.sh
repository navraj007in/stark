#!/bin/bash

# Set root directory name
ROOT_DIR="."

# Create directory structure
mkdir -p $ROOT_DIR/docs/{00-Overview,01-Architecture,02-Type-System,03-Syntax,04-Concurrency,05-Cloud-Native,06-Standard-Library,07-Compiler-Toolchain,08-Runtime,09-Examples}
mkdir -p $ROOT_DIR/design-diagrams

# Create placeholder markdown files
touch $ROOT_DIR/README.md

# Overview
touch $ROOT_DIR/docs/00-Overview/{Mission-Statement.md,Vision.md,Feature-Roadmap.md}

# Architecture
touch $ROOT_DIR/docs/01-Architecture/{Execution-Model.md,STARKVM.md,JIT-Compiler.md}

# Type System
touch $ROOT_DIR/docs/02-Type-System/{Primitive-Types.md,Composite-Types.md,AI-Types.md,Type-Inference.md}

# Syntax
touch $ROOT_DIR/docs/03-Syntax/{Basic-Syntax.md,Control-Structures.md,Functions-Modules.md}

# Concurrency
touch $ROOT_DIR/docs/04-Concurrency/{Actor-System.md,Async-Await.md,Parallel-Patterns.md}

# Cloud Native
touch $ROOT_DIR/docs/05-Cloud-Native/{Serverless-Support.md,Deployment-Primitives.md}

# Standard Library
touch $ROOT_DIR/docs/06-Standard-Library/{TensorLib.md,DatasetLib.md,Networking.md}

# Compiler Toolchain
touch $ROOT_DIR/docs/07-Compiler-Toolchain/{Bytecode-Format.md,Compiler-Stages.md,Dev-Tooling.md}

# Runtime
touch $ROOT_DIR/docs/08-Runtime/{Memory-Management.md,Garbage-Collector.md}

# Examples
touch $ROOT_DIR/docs/09-Examples/{Hello-World.stark,ML-Pipeline.stark}

# Index
touch $ROOT_DIR/docs/index.md

# Initialize git repo
cd $ROOT_DIR
git init > /dev/null
cd ..

# Zip it all up
zip -r "${ROOT_DIR}.zip" $ROOT_DIR > /dev/null

echo "📁 STARK Language doc structure created and zipped as ${ROOT_DIR}.zip"
