function parabel_train( trn_X_Xf, trn_X_Y, model_dir, param )

    file_trn_X_Xf = tempname;
    write_text_mat( trn_X_Xf, file_trn_X_Xf );
    
    file_trn_X_Y = tempname;
    write_text_mat( trn_X_Y, file_trn_X_Y );

    cmd = sprintf('parabel_train %s %s %s %s', file_trn_X_Xf, file_trn_X_Y, model_dir, get_arguments(param));
    if isunix
       cmd = ['./' cmd]; 
    end
	disp( cmd );
    system(cmd);
end

function args = get_arguments(param)
	args = ' ';

	if isfield(param,'num_thread')
		args = sprintf(' %s -T %d',args,param.num_thread);
	end

    if isfield(param,'start_tree')
		args = sprintf(' %s -s %d',args,param.start_tree);
    end
    
	if isfield(param,'num_tree')
		args = sprintf(' %s -t %d',args,param.num_tree);
	end

	if isfield(param,'bias_feat')
		args = sprintf(' %s -b %f',args,param.bias_feat);
	end

	if isfield(param,'classifier_cost')
		args = sprintf(' %s -c %f',args,param.classifier_cost);
	end

	if isfield(param,'max_leaf')
		args = sprintf(' %s -m %d',args,param.max_leaf);
	end

	if isfield(param,'classifier_threshold')
		args = sprintf(' %s -tcl %f',args,param.classifier_threshold);
	end

	if isfield(param,'centroid_threshold')
		args = sprintf(' %s -tce %f',args,param.centroid_threshold);
	end

	if isfield(param,'clustering_eps')
		args = sprintf(' %s -e %f',args,param.clustering_eps);
    end

	if isfield(param,'classifier_maxitr')
		args = sprintf(' %s -n %d',args,param.classifier_maxitr);
	end

    if isfield(param,'classifier_kind')
        args = sprintf(' %s -k %d',args,param.classifier_kind);
    end

	if isfield(param,'quiet')
		args = sprintf(' %s -q %d',args,param.quiet);
    end

end
