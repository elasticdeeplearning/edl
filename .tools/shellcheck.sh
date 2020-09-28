# https://stackoverflow.com/questions/6879501/filter-git-diff-by-type-of-change
changed_sh_files=$(git diff --cached --name-only --diff-filter=ACMR | grep '\.sh$')
if [[ "${changed_sh_files}" == "" ]]; then
    exit 0
fi

echo "${changed_sh_files}" | xargs shellcheck
