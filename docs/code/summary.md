# Summary Backend

PEAX has the ability to create ```summaries```.
These are HTML-based files that contain the information that is contained within the ModelAnalysis, reports and rewrites.

This functionality should be implemented in your custom reports and rewriters through their ```render_summary``` function, which is supposed to create the html file(s) and write them to the given folder (using a unique name), it returns the title and file_path of the main file to the calling function, to enable the ModelAnalysis object to link all summary files created by the reports and rewriters, it contains.