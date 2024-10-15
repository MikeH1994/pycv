import matplotlib.pyplot as plt


class Metric:
    default_x_label = None
    default_y_label = None

    def plot(self, **kwargs):
        new_figure = kwargs["new_figure"] if "new_figure" in kwargs else True
        if new_figure:
            plt.figure()

        overwrite = kwargs["overwrite"] if "overwrite" in kwargs else True
        self.plot_elem(**kwargs)

        show = kwargs["show"] if "show" in kwargs else False
        legend = kwargs["legend"] if "legend" in kwargs else False
        title = kwargs["title"] if "title" in kwargs else None
        xlabel = kwargs["xlabel"] if "xlabel" in kwargs else self.default_x_label
        ylabel = kwargs["ylabel"] if "ylabel" in kwargs else self.default_y_label

        if overwrite:
            if title is not None:
                plt.title(title)
            if xlabel is not None:
                plt.xlabel(xlabel)
            if ylabel is not None:
                plt.ylabel(ylabel)
        if "xlim" in kwargs:
            plt.xlim(kwargs["xlim"])
        if "ylim" in kwargs:
            plt.ylim(ylim)
        if legend:
            plt.legend(loc=0)
        if show:
            plt.show()

    def plot_elem(self, **kwargs):
        raise Exception("Base function Metric.plot_elem() called")
