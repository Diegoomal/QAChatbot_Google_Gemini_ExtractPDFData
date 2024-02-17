
echo "===== bash destroy ====="

echo "1) Remove conda env"

conda deactivate

conda remove --name project-env --all -y

echo "2) Remove dir and files"

rm -rf __pycache__
rm -rf .pytest_cache

rm -rf src/__pycache__

rm -rf docs

rm -rf tests/__pycache__