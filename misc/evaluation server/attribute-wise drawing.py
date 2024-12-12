result_plot_path = r"" # the only thing you need to configure
import os
import sys
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import torch
import tikzplotlib
plot_draw_style = [{'color': (1.0, 0.0, 0.0), 'line_style': '-'},
                       {'color': (0.0, 1.0, 0.0), 'line_style': '-'},
                       {'color': (0.0, 0.0, 1.0), 'line_style': '-'},
                       {'color': (0.0, 0.0, 0.0), 'line_style': '-'},
                       {'color': (1.0, 0.0, 1.0), 'line_style': '-'},
                       {'color': (0.0, 1.0, 1.0), 'line_style': '-'},
                       {'color': (0.5, 0.5, 0.5), 'line_style': '-'},
                       {'color': (136.0 / 255.0, 0.0, 21.0 / 255.0), 'line_style': '-'},
                       {'color': (1.0, 127.0 / 255.0, 39.0 / 255.0), 'line_style': '-'},
                       {'color': (0.0, 162.0 / 255.0, 232.0 / 255.0), 'line_style': '-'},
                       {'color': (0.0, 0.5, 0.0), 'line_style': '-'},
                       {'color': (1.0, 0.5, 0.2), 'line_style': '-'},
                       {'color': (0.1, 0.4, 0.0), 'line_style': '-'},
                       {'color': (0.6, 0.3, 0.9), 'line_style': '-'},
                       {'color': (0.4, 0.7, 0.1), 'line_style': '-'},
                       {'color': (0.2, 0.1, 0.7), 'line_style': '-'},
                       {'color': (0.7, 0.6, 0.2), 'line_style': '-'},
                       {'color': (255.0 / 255.0, 102.0 / 255.0, 102.0 / 255.0), 'line_style': '-'},
                       {'color': (153.0 / 255.0, 255.0 / 255.0, 153.0 / 255.0), 'line_style': '-'},
                       {'color': (102.0 / 255.0, 102.0 / 255.0, 255.0 / 255.0), 'line_style': '-'},
                       {'color': (255.0 / 255.0, 192.0 / 255.0, 203.0 / 255.0), 'line_style': '-'},
                       {'color': (255.0 / 255.0, 215.0 / 255.0, 0 / 255.0), 'line_style': '-'}
                       ]
attribution_list = ['Occlusion', 'Out Of View', 'Camera Motion', 'Deformation', 'Fast Motion', 'Low Discriminative', 'Motion Blur', 'Tiny Target']

success_plot_opts = {'plot_type': 'success', 'legend_loc': 'lower left', 'xlabel': 'Overlap threshold',
                             'ylabel': 'Overlap Precision [%]', 'xlim': (0, 1.0), 'ylim': (0, 100), 'title': 'Success plot'}
precision_plot_opts = {'plot_type': 'precision', 'legend_loc': 'lower right',
                               'xlabel': 'Location error threshold', 'ylabel': 'Distance Precision [%]',
                               'xlim': (0, 50), 'ylim': (0, 100), 'title': 'Precision plot'}



def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            data = data['attribution']
            data = data.replace("'", '"')
            data = json.loads(data)
        return data
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"The file {file_path} is not a valid JSON file.")
        return None

def plot_draw_save(y, x, scores, trackers, plot_draw_styles, result_plot_path, plot_opts,att,frames):
    if att == 'Low Discriminative':
        att = "Distractors"
    # Plot settings
    font_size = plot_opts.get('font_size', 12)
    font_size_axis = plot_opts.get('font_size_axis', 13)
    line_width = plot_opts.get('line_width', 2)
    font_size_legend = plot_opts.get('font_size_legend', 8)

    plot_type = plot_opts['plot_type']
    legend_loc = plot_opts['legend_loc']

    xlabel = plot_opts['xlabel']
    ylabel = plot_opts['ylabel']
    xlim = plot_opts['xlim']
    ylim = plot_opts['ylim']

    title = plot_opts['title'] + " on " + att + " " + "(" + str(frames) + ")"

    matplotlib.rcParams.update({'font.size': font_size})
    matplotlib.rcParams.update({'axes.titlesize': font_size_axis})
    matplotlib.rcParams.update({'axes.titleweight': 'black'})
    matplotlib.rcParams.update({'axes.labelsize': font_size_axis})

    fig, ax = plt.subplots()

    index_sort = scores.argsort(descending=False)

    plotted_lines = []
    legend_text = []
    #print(y[index_sort[0], :])
    for id, id_sort in enumerate(index_sort):
        line = ax.plot(x, y[id_sort, :].tolist()[0],
                       linewidth=line_width,
                       color=plot_draw_styles[index_sort.numel() - id - 1]['color'],
                       linestyle=plot_draw_styles[index_sort.numel() - id - 1]['line_style'])

        plotted_lines.append(line[0])

        tracker = trackers[id_sort]
        disp_name = tracker

        legend_text.append('{} [{:.1f}]'.format(disp_name, scores[id_sort]))

    ax.legend(plotted_lines[::-1], legend_text[::-1], loc=legend_loc, fancybox=False, edgecolor='black',
              fontsize=font_size_legend, framealpha=1.0)

    ax.set(xlabel=xlabel,
           ylabel=ylabel,
           xlim=xlim, ylim=ylim,
           title=title)

    ax.grid(True, linestyle='-.')
    fig.tight_layout()

    tikzplotlib.save('{}/{}_plot.tex'.format(result_plot_path, plot_type))
    fig.savefig('{}/{}_plot.png'.format(result_plot_path, att+"_"+plot_type), dpi=300, format='png', transparent=True)
    plt.draw()



files = os.listdir('.')


json_files = [file for file in files if file.endswith('.json')]


cleaned_files = [os.path.splitext(file)[0] for file in json_files]
if len(json_files) > len(plot_draw_style):
    print(f"too much trackers, make sure the json file is less than {len(plot_draw_style)}")
    sys.exit()

x1 = read_json_file(json_files[0])['Occlusion']['x'][0]
x2 = read_json_file(json_files[0])['Occlusion']['x'][1]


for att in attribution_list:
    y1 = []
    y2 = []
    
    z1 = []
    z2 = []
    for jsonfile in json_files:
        y1.append(np.array(read_json_file(jsonfile)[att]['y'][0]))
        y2.append(np.array(read_json_file(jsonfile)[att]['y'][1]))
        z1.append(np.array(read_json_file(jsonfile)[att]['z'][0]))
        z2.append(np.array(read_json_file(jsonfile)[att]['z'][1]))
    
    y1 = np.array(y1)
    y2 = np.array(y2)
    z1 = np.array(z1)
    z2 = np.array(z2)
    total_frames = np.array(read_json_file(jsonfile)[att]['frames'])
    
    
    y1= torch.tensor(y1)
    y2= torch.tensor(y2)
    z1= torch.tensor(z1)
    z2= torch.tensor(z2)
    
    plot_draw_save(y1, x1, z1, cleaned_files, plot_draw_style, result_plot_path, success_plot_opts,att,total_frames)
    
    
    plot_draw_save(y2, x2, z2, cleaned_files, plot_draw_style, result_plot_path, precision_plot_opts,att,total_frames)