# extract the last version from folder: ogs/CrossNet_$SLURM_JOB_NAME/version_1/checkpoints/last.ckpt 

# Usage: extract_version.sh <path_to_folder>
# Example: extract_version.sh logs/CrossNet_$SLURM_JOB_NAME/version_1/checkpoints/last.ckpt

# Output: version number

VERSION=$(ls $1 | grep -oP '(?<=version_)\d+' | sort -n | tail -1)
echo version_${VERSION}