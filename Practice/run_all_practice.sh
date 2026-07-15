#!/bin/bash
set -e

# Path to compiler binary
STARKC="../starkc/target/debug/starkc"

echo "=== 1. Building compiler ==="
cd ../starkc
cargo build
cd ../Practice

echo "=== 2. Checking all conforming files ==="
$STARKC check Conforming/01_basics.stark
$STARKC check Conforming/02_error_handling.stark
$STARKC check Conforming/03_ownership.stark
$STARKC check Conforming/04_traits_generics.stark
$STARKC check --extension tensor Conforming/05_tensors.stark
$STARKC check Conforming/06_file_io.stark

echo "=== 3. Checking all algorithm files ==="
$STARKC check Algorithms/01_binary_search.stark
$STARKC check Algorithms/02_bubble_sort.stark
$STARKC check Algorithms/03_stack.stark
$STARKC check Algorithms/04_linked_list.stark
$STARKC check Algorithms/05_binary_tree.stark
$STARKC check Algorithms/06_graph.stark

echo "=== 4. Executing conforming examples ==="
$STARKC run Conforming/01_basics.stark
$STARKC run Conforming/02_error_handling.stark
$STARKC run Conforming/03_ownership.stark
$STARKC run Conforming/04_traits_generics.stark
$STARKC run Conforming/06_file_io.stark

echo "=== 5. Executing algorithm examples ==="
$STARKC run Algorithms/01_binary_search.stark
$STARKC run Algorithms/02_bubble_sort.stark
$STARKC run Algorithms/03_stack.stark
$STARKC run Algorithms/04_linked_list.stark
$STARKC run Algorithms/05_binary_tree.stark
$STARKC run Algorithms/06_graph.stark

echo ""
echo "=================================================="
echo "Success: All practice programs checked and executed!"
echo "=================================================="
