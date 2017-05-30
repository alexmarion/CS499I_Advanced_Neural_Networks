function [ dirs ] = remove_dot_dirs( dirs )
%remove_dot_dirs removes '.', '..', and '.DS_Store'
    dirs = dirs(~strncmpi('.', {dirs.name}, 1));
end

