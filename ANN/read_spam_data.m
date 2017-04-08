function [ index, fields ] = read_spam_data()

    filename = 'spambase.data';
    datafile = 'spambase.mat';
    
    if(exist(datafile,'file'))
        load(datafile);
    else
        fid = fopen(filename);
        if(fid < 0)
            display('File not found');
            return;
        end

        index = [];
        fields = [];

        line = fgetl(fid);  % read the first line to get D
        D = length(strfind(line,','));  % Number of features not including index
        
        format = strcat('%d,',repmat('%f,', 1, D-1),'%f');

        fid = fopen(filename);  % reset fid to the first line

        while(~feof(fid))
            line = fgetl(fid);  % read the next line
            C = textscan(line,format);
            
            t = [];
            for i = 1:D
                t(1, end+1) = C{i}; % Collect each field in a temp cell array
            end

            index(end+1,1) = C{end};
            fields(end+1, :) = t;
        end
        fclose(fid);
        save(datafile,'index','fields');
    end
end

