import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

#from IPython.core.debugger import set_trace

# Some colormaps chosen to mirror the default (Cx) series of colors
colortraces = [plt.matplotlib.cm.Blues,
              plt.matplotlib.cm.YlOrBr, # This in order to use oranges for the brouwns
              plt.matplotlib.cm.Greens,
              plt.matplotlib.cm.Reds,
              plt.matplotlib.cm.Purples,
              plt.matplotlib.cm.Oranges,
              plt.matplotlib.cm.RdPu,
              plt.matplotlib.cm.Greys,
              plt.matplotlib.cm.YlOrRd,
              plt.matplotlib.cm.GnBu]


def lambdifyz(symbols, expr, modules="numpy"):
    """
    Circumvents a bug in lambdify where silent 
    variables will be simplified and therefore
    aren't broadcasted. https://github.com/sympy/sympy/issues/5642
    Use an extra argument = 0 when calling
    """
    assert isinstance(expr, sp.Matrix)
    z = sp.symbols("z")
    thesymbols = list(symbols)
    thesymbols.append(z)
    exprz = expr + z*sp.prod(symbols)*sp.ones(expr.shape[0], expr.shape[1])
    fz = sp.lambdify(thesymbols, exprz, modules=modules)
    return fz


def plotitem(axs, item, plotted, nx, idx,k, osfrac=0.1,verbose=False,
             baseoffset=0, linestyle="-", label="X", linewidth=5,
             labels=True, projection="polar", rmax=1.,zorder=1):
    """
    A function that serves as a macro to plot the complex amplitude vectord for CMP
    
    axs      : The axis objects for the plot
    item     : The complex amplitude to plot
    plotted  : The complex amplitude already plotted (things will get staggered if 
                    two identical complex amplitudes are plotted)
    nx       : The number of columns of plots (indexing is 1D if there is only one column)
    i        : The first index of the plot
    j        : The second index of the plot
    k        : The index of the phasor to plot
    osfrac   : The fraction of the amplitude to use as offset
    verbose  : Gives more information on what happens
    baseoffset : The offset in the start of the vector to use (only special cases)
    linestyle : Used for plotting dashed lines
    label    : The label to use for the legend
    linewidth : Self explanatory
    labels   : Whether to include a little label for each vector
    projection : Whether to use a polar or cartesian projection (cartesian no longer maintained)
    rmax      : The maximum norm to for the plot 
    """
    
    offset = osfrac*np.abs(item)*np.exp(1j*(np.angle(item)+np.pi/2))
    
    while item in plotted:
        item += offset
        baseoffset += offset
    if projection=="polar":
        a0=np.angle(baseoffset)
        b0=np.abs(baseoffset)
        a1=np.angle(item)
        b1=np.abs(item)
        a2=np.angle(offset)
        b2=np.abs(offset)
    else:
        a0=np.real(baseoffset)
        b0=np.imag(baseoffset)
        a1=np.real(item)
        b1=np.imag(item)
        a2=np.real(offset)
        b2=np.imag(offset)
    if nx==1:
        #axs[i,j].scatter(matrix[i*nx+j,k].real, matrix[i*nx+j,k].imag)
        axs[idx].plot([a0,a1], [b0,b1],
                      color="C"+str(k), linewidth=linewidth,
                      linestyle=linestyle, label=label, zorder=zorder)
        if labels:
            axs[i].text(0.95*a1, 0.9*b1, str(k))
        axs[idx].set_aspect("equal")
        axs[idx].set_ylim(0,rmax)

    else:

        #axs[i,j].scatter(matrix[i*nx+j,k].real, matrix[i*nx+j,k].imag)
        axs[idx].plot([a0,a1], [b0,b1],
                      color="C"+str(k), linewidth=linewidth,
                      linestyle=linestyle, label=label, zorder=zorder)
        if labels:
            axs[idx].text(0.95*a1, 0.9*b1, str(k))
        axs[idx].set_aspect("equal")
        axs[idx].set_ylim(0,rmax)
    plotted.append(item)
    return plotted

def plotitem_arrow(axs, item, plotted, nx, idx,k, osfrac=0.1,verbose=False,
             baseoffset=0, linestyle="-", label="X", linewidth=0.025,
             labels=True, projection="polar", rmax=1., addring=False, zorder=1):
    """
    A function that serves as a macro to plot the complex amplitude vectord for CMP
    
    axs      : The axis objects for the plot
    item     : The complex amplitude to plot
    plotted  : The complex amplitude already plotted (things will get staggered if 
                    two identical complex amplitudes are plotted)
    nx       : The number of columns of plots (indexing is 1D if there is only one column)
    i        : The first index of the plot
    j        : The second index of the plot
    k        : The index of the phasor to plot
    osfrac   : The fraction of the amplitude to use as offset
    verbose  : Gives more informations on what happens
    baseoffset : The offset in the start of the vector to use (only special cases)
    linestyle : Used for plotting dashed lines
    label    : The label to use for the legend
    linewidth : Currently not corrected (the scale is very different from the other plotitem function)
    labels   : Whether to include a little label for each vector
    projection : Whether to use a polar or cartesian projection (cartesian not tested)
    rmax      : The maximum norm to for the plot 
    """
    
    offset = osfrac*np.abs(item)*np.exp(1j*(np.angle(item)+np.pi/2))
    if k == "black":
        thecolor = "k"
    else:
        thecolor = "C"+str(k)
    
    if verbose: print("initial",item)
    while item+baseoffset in plotted:
        #item += offset
        baseoffset += offset
        if verbose: print("shifting")
    if projection=="polar":
        a0=np.angle(baseoffset)
        if verbose: print("final, base angle", a0)
        b0=np.abs(baseoffset)
        if verbose: print("final, base norm", b0)
        a1=np.angle(item+baseoffset) - np.angle(baseoffset)
        if verbose: print("final, item angle", a1)
        b1=np.abs(item + baseoffset) - np.abs(baseoffset)
        if verbose: print("final, item norm", b1)
        a2=np.angle(offset)
        if verbose: print("final, offset angle", a2)
        b2=np.abs(offset)
        if verbose: print("final, offset norm", b2)
    else:
        a0=np.real(baseoffset)
        b0=np.imag(baseoffset)
        a1=np.real(item)
        b1=np.imag(item)
        a2=np.real(offset)
        b2=np.imag(offset)
    if nx==1:
        #axs[i,j].scatter(matrix[i*nx+j,k].real, matrix[i*nx+j,k].imag)
        axs[idx].quiver(a0, b0, a1, b1, scale_units='xy', angles='xy', scale=1,
                      color=thecolor, width=linewidth, headlength=2.5, headaxislength=2.2,
                      linestyle=linestyle, label=label, zorder=zorder)
        if labels:
            axs[idx].text(0.95*a1, 0.9*b1, str(k))
        axs[idx].set_aspect("equal")
        axs[idx].set_ylim(0,rmax)
        if addring:
            thetas = np.linspace(0,2*np.pi,100)
            axs[idx].plot(thetas, np.ones_like(thetas)*np.abs(item),
                                      color=thecolor, zorder=zorder)

    else:

        #axs[idx].scatter(matrix[i*nx+j,k].real, matrix[i*nx+j,k].imag)
        axs[idx].quiver(a0, b0, a1, b1, scale_units='xy', angles='xy', scale=1,
                      color=thecolor, width=linewidth, headlength=2.5, headaxislength=2.2,
                      linestyle=linestyle, label=label, zorder=zorder)
        if labels:
            axs[idx].text(0.95*a1, 0.9*b1, str(k))
        axs[idx].set_aspect("equal")
        axs[idx].set_ylim(0,rmax)
        if addring:
            thetas = np.linspace(0,2*np.pi,100)
            axs[idx].plot(thetas, np.ones_like(thetas)*np.abs(item),
                                      color=thecolor, zorder=zorder)
    plotted.append(item+baseoffset)
    return plotted


def plot_outputs_smart(matrix=None, inputfield=None, base_preoffset=None, nx=2,ny=None,legendoffset=(-0.2,0),#(1.6,0.5),
                       verbose=False, osfrac=0.1, plotsize=3, plotspaces=(0.3,0.4), onlyonelegend=False,
                       labels=False, legend=True,legendsize=8, legendstring="center left", title=None, projection="polar",
                       out_label=None, rmax=None, show=True, onlyoneticklabel=False, labelsize=15,
                       rlabelpos=20, autorm=False, plotter=plotitem_arrow, mainlinewidth=0.04, outputontop=False,
                       thealpha=0.1, color=("black", "silver"), outlabelloc=None, dpi=100):
        """
        Produces a Complex Matrix Plot (CMP) of a combiner matrix. The matrix represents the phasors in each cell of the matrix. In cases where the matrix is designed to take as an input cophased beams of equal amplitude, the plots can also be seen as a representation of the decomposition of the outputs into the contribution of each input.
        returns a fig, and  axs objects.
        matrix   : The matrix to plot.
        inputfield: An input field fed to the combiner. If provided, a plot of M.dot(np.dag(inputfield))
        nx       : The number of columns of plots
        ny       : The number of rows of plots (if provided)
        legendoffset: An offset for the legend (to put it in a free space)
        verbose  : 
        osfrac   : The fraction of the amplitude of phasors to use as offset.
        plotsize : The size (matplotlib inches) of each plot
        plotspaces: The space between plots
        labels   : Whether to use the little numbers on each phasor
        legend   : Whether to use a legend for the colors
        legendsize: The size of the legend
        legendstring: The string used by pyplot to use as reference for the legend location
        title    : A title for the whole figure: either the title string, a None object (use default title) or a False boolean (no title).
        projection: if "polar" will use polar plots
        out_label: colored labels for each of the output plot. Requires an array corresponding to each row of the matrix.
        thealpha : Alpha for output label box
        rmax     : The outer limit of the plot (max amplitude)
        show     : Whether to plt.show the figure at the end.
        onlyoneticklabel: Remove tick labels for all but the bottom left plots
        labelsize: Size fot the tick labels (default: 15))
        rlabelpos: The angle at which to put the amplitude tick labels (default: 20)
        autorm   : Automatically remove empty rows True=auto, False=keep all, Boolean array: the rows to remove
        plotter  : A function for plotting fancy arrow vectors
        """
        special = True
        
            
        if inputfield is not None:
            initialmatrix = matrix
            matrix = matrix.dot(np.diag(inputfield))
            outvec = initialmatrix.dot(inputfield)
        else :
            initialmatrix = np.zeros((0,0))
            outvec = None
        if ny is None:
            ny = matrix.shape[0]//nx
        if matrix.shape[0]%nx != 0:
            ny = ny +1
        ntot = matrix.shape[0]
        
        if projection=="polar":
            sharex="none"
            sharey="all"
            text_coords="polar"
        else:
            sharex="all"
            sharey="all"
            text_coords="data"
        if rmax is None:    
            rmax=np.max(matrix)
        if base_preoffset is None:
            base_preoffset = np.zeros_like(matrix)
        
        fig, axs = plt.subplots(ny,nx,sharex=sharex, sharey=sharey,
                                gridspec_kw={'hspace': plotspaces[0], 'wspace': plotspaces[1]},
                                figsize=(plotsize*nx,plotsize*matrix.shape[0]//nx+0.5),
                                subplot_kw=dict(projection=projection), dpi=dpi)
        
        for idx, theax in enumerate(axs.flatten()):
                if (idx==0) or (not labels):
                    addlabel=False
                else:
                    addlabel=True
                
                #Plotting the output result (black stuff) on the bottom!
                if (outvec is not None) and ((idx)<matrix.shape[0]) and not outputontop:
                    plotted=[]
                    baseoffset = 0
                    item = outvec[idx]
                    plotted = plotter(axs.flat, item, plotted, nx, idx, "black", verbose=verbose,
                                           osfrac=osfrac, baseoffset=baseoffset,linewidth=mainlinewidth,
                                           linestyle="-", label="Output "+str(idx), labels=addlabel,
                                           projection=projection, rmax=rmax, addring=True, zorder=1)
                    
                    
                    
                plotted = []
                adjust=[]
                for k in range(matrix.shape[1]):
                    if (idx)<matrix.shape[0]:
                        item = matrix[idx,k] # base_preoffset[idx,k]
                        baseoffset = base_preoffset[idx,k]
                        if item==0:
                            continue
                        #Here we use plotter, the optional function for plotting vectors
                        plotted = plotter(axs.flat, item, plotted, nx, idx, k, verbose=verbose,
                                           osfrac=osfrac, baseoffset=baseoffset,linewidth=mainlinewidth,
                                           linestyle="-", label="Input "+str(k), labels=addlabel,
                                           projection=projection, rmax=rmax)
                
                plotted2 = []
                adjus2t = []
                #Plotting the dashed lines for the matrix itself
                for k in range(initialmatrix.shape[1]):
                    if (idx)<initialmatrix.shape[0]:
                        item = initialmatrix[idx,k]
                        baseoffset = 0
                        if item==0:
                            continue
                        if idx != 0:#We just don't plot the reference for the bright
                            plotted = plotitem(axs.flat, item, plotted2, nx, idx, k,
                                           osfrac=osfrac, baseoffset=baseoffset,
                                           linestyle="--", label=None, labels=False,
                                           projection=projection, rmax=rmax, linewidth=3, zorder=0)
                #Plotting the output result (black stuff)
                if (outvec is not None) and ((idx)<matrix.shape[0]) and outputontop:
                    print("we do plot on top")
                    baseoffser = 0
                    plotted=[]
                    item = outvec[idx]
                    plotted = plotter(axs.flat, item, plotted, nx, idx, "black", verbose=verbose,
                                           osfrac=osfrac, baseoffset=baseoffset,linewidth=mainlinewidth,
                                           linestyle="-", label="Output", labels=addlabel,
                                           projection=projection, rmax=rmax, addring=True, zorder=4)
                    
                
                    
                if legend:
                    if onlyonelegend:
                        if idx ==0:
                            axs.flat[idx].legend(loc=legendstring, prop={'size': legendsize}, bbox_to_anchor=legendoffset)
                    else:
                        axs.flat[idx].legend(loc=legendstring, prop={'size': legendsize}, bbox_to_anchor=legendoffset)
                            
                if out_label is not None:
                    #set_trace()
                    if outlabelloc is None:
                        outlabelloc = (1.18*np.pi/2,1.32*rmax)
                    for idx, theax in enumerate(axs.flatten()):
                        #print(theax)
                        if color is None:
                            edgecolor = "C"+str((idx)//2)
                            facecolor = "C"+str((idx)//2)
                        elif color is False:
                            facecolor = "white"
                            edgecolor = "black"
                        else:
                            edgecolor, facecolor = color

                        theax.text(outlabelloc[0], outlabelloc[1], str(out_label[idx]), size=15,
                                ha="right", va="top", 
                                bbox=dict(boxstyle="round",
                                #facecolor="none",
                                alpha=thealpha,
                                facecolor=facecolor,
                                edgecolor=edgecolor,
                                ))
                    
        #eliminating the empty plots
        if autorm is True:
            rowstoremove = np.prod(matrix, axis=1) == 0
        elif autorm is False:
            rowstoremove = np.zeros(matrix.shape[0], dtype=np.bool)
        else :
            rowstoremove = autorm
        #Making a pretty legend for the phase term
        xT = np.arange(0, 2*np.pi,np.pi/4)
        #xL=['0',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',\
        #        r'$\pi$',r'$\frac{5\pi}{4}$',r'$\frac{3\pi}{2}$',r'$\frac{7\pi}{4}$']
        #xL=['0',r'$\pi/4$',r'$\pi/2$',r'$3\pi/4$',\
        #        r'$\pi$',r'$5\pi/4$',r'$3\pi/2}$',r'$7\pi/4$']
        xT = np.arange(0, 2*np.pi,np.pi/2)
        xL=['0',r'$\pi/2$', r'$\pi$',r'$3\pi/2}$']
        #copying teh ytick labels from the first plot
        #yT = np.linspace(0,0.75*rmax.real,3).round(decimals=1)[1:]
        #yL = [str(yT[b]) for b in np.arange(yT.shape[0])]
        removeticklabels = np.zeros_like(rowstoremove)
        if onlyoneticklabel:
            removeticklabels = np.ones_like(rowstoremove)
            removeticklabels[-nx] = 0
        print("removing the labels:", removeticklabels)
        #print("removing the rows:", rowstoremove)
        
        for i in np.flip(np.arange(matrix.shape[0])):
            
            fig.axes[i].set_xticks(xT)
            fig.axes[i].set_xticklabels(xL)
            #fig.axes[i].set_rgrids(yT,yL)
            fig.axes[i].yaxis.set_tick_params(labelbottom=True)
                
            fig.axes[i].set_rlabel_position(rlabelpos)
            fig.axes[i].tick_params(labelsize=labelsize)
            #print("adding labels",xL)
            if rowstoremove[i] :
                fig.axes[i].remove()
            if removeticklabels[i]:
                fig.axes[i].set_xticklabels([])
                #fig.axes[i].set_yticklabels([])
                            
        if title is not False:
            if title is None:
                title = "The null configurations\n of all the %d outputs"%(matrix.shape[0])
            fig.suptitle(title)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, axs


####################################
# For plotting chromatic combiners


from pdb import set_trace
def plot_trace(axis, item, wl, cmap, label=None, verbose=False, msize=10.,
              rmax=None, cbar_ax=None, k=None,minfrac=0.9, maxfrac=None, **kwargs):
    r = np.abs(item)
    phi = np.angle(item)
    if verbose:
        print("r", r)
        print("phi", phi)
        print("c", c)
    if maxfrac is None:
        maxfrac=1.
    if minfrac is None:
        minfrac=1.
    cmin = minfrac * np.min(wl)
    cmax = maxfrac * np.max(wl)
    axis.scatter(phi, r, label=label, c=wl, cmap=cmap,
                 vmin=cmin, vmax=cmax, **kwargs)
    axis.set_aspect("equal")
    axis.set_ylim(0,rmax)
    if cbar_ax is not None:
        cbar_ax.scatter(wl, k*np.ones_like(wl), c=wl, cmap=cmap,
                        vmin=cmin, vmax=cmax, **kwargs)
    
    
    
    
    
    
    
    
def plot_chromatic_matrix(amatrix, lamb, wlpoints,nx=2,ny=None,legendoffset=(-0.2,0),#(1.6,0.5),
               verbose=False, plotsize=3, plotspaces=(0.3,0.4),
               plotout=False,minfrac=None,maxfrac=None,
               labels=False, legend=True,legendsize=8, legendstring="center left", title=None,
               out_label=None, rmax=None, show=True, labelsize=15,colors=colortraces,
               rlabelpos=45, autorm=False, plotter=plot_trace,
               mainlinewidth=0.04, outputontop=True, msize=10,
               thealpha=0.5, color=("black", "silver"), outlabelloc=None, dpi=100,
               returnmatrix=False):
        matfunction = lambdifyz((lamb,), amatrix, modules="numpy")
        broacasted = matfunction(wlpoints, 0)
        #matfunction = lambdify_void_mat((lamb,), amatrix, modules="numpy")
        #broacasted = matfunction(wlpoints)#.astype(np.complex128)
        matrix = np.moveaxis(broacasted, 2, 0)
        if verbose:print(matrix)
        if ny is None:
            ny = matrix.shape[1]//nx
        if matrix.shape[1]%nx != 0:
            ny = ny +1
        ntot = matrix.shape[2]
        
        projection = "polar"
        sharex="none"
        sharey="all"
        text_coords="polar"
        if rmax is None:    
            rmax=np.max(np.abs(matrix))
        
        fsize = (plotsize*nx,plotsize*matrix.shape[1]//nx+0.5)
        print(fsize)
        
        fig, axs = plt.subplots(ny,nx,sharex=sharex, sharey=sharey,
                                gridspec_kw={'hspace': plotspaces[0], 'wspace': plotspaces[1]},
                                figsize=fsize,
                                subplot_kw=dict(projection="polar"), dpi=dpi)
        
        fig.subplots_adjust(top=0.9)
        cbar_ax = fig.add_axes([0.05, 0.95, 0.9, 0.1])
        plt.xlabel("Wavelength [m]")
        plt.ylabel("Input index (Grey + is output)")
        plt.title(title)
        
        if plotout is not False:
            if isinstance(plotout, sp.Matrix):
                inarray = plotout
                output = amatrix@inarray
                out_matfunction = lambdifyz((lamb,), output, modules="numpy")
                out_broacasted = out_matfunction(wlpoints, 0)
                out_vector =  np.moveaxis(out_broacasted, 2, 0)
                in_matfunction = lambdifyz((lamb,), inarray, modules="numpy")
                in_broacasted = in_matfunction(wlpoints, 0)
                matrix = np.einsum("ikj,ij->ikj", matrix, in_broacasted)
            elif isinstance(plotout, np.ndarray):
                inarray = plotout
                out_matfunction = lambdifyz((lamb,), amatrix, modules="numpy")
                out_broacasted = out_matfunction(wlpoints, 0)
                out_switch =  np.moveaxis(out_broacasted, 2, 0)
                print("out_switch", out_switch.shape)
                print("inarray", inarray.shape)
                print("")
                out_vector =  np.einsum("ikj,ij->ik",
                                        out_switch, inarray)
                #out_vector = inarray
                matrix = np.einsum("ikj,ij->ikj", matrix, plotout)
                
            else:
                inarray = sp.ones(amatrix.shape[1],1)
                output = amatrix@inarray
                out_matfunction = lambdifyz((lamb,), output, modules="numpy")
                out_broacasted = out_matfunction(wlpoints, 0)
                out_vector =  np.moveaxis(out_broacasted, 2, 0)
            #out_matfunction = lambdify_void_mat((lamb,), output, modules="numpy")
            #out_broacasted = out_matfunction(wlpoints)
            #out_matfunction = ufuncify(lamb, output, backend="numpy")
            #out_broacasted = out_matfunction(wlpoints)
            
            for idx, theax in enumerate(axs.flatten()):
                item = out_vector[:,idx]
                #print(item)
                plotted = plotter(theax, item, wlpoints,
                                  cmap="Greys",
                                  label="Output "+str(idx),
                                  verbose=verbose,msize=msize, rmax=rmax,
                                  cbar_ax=cbar_ax, k=ntot, alpha=thealpha,
                                  marker="+", s=100.,
                                  minfrac=minfrac,
                                  maxfrac=maxfrac)
        
        for idx, theax in enumerate(axs.flatten()):
            plotted = []
            for k in range(matrix.shape[2]):
                if (idx)<matrix.shape[1]:
                    item = matrix[:,idx,k] # base_preoffset[idx,k]
                    cmap = colors[k]
                    if verbose: print(item)
                    if not np.allclose(item, 0): #Avoid plotting 0 elements
                        #Here we use plotter, the optional function for plotting vectors
                        plotted = plotter(axs.flat[idx], item, wlpoints, cmap=cmap, label="Input "+str(k),
                                           verbose=verbose,msize=msize, rmax=rmax,
                                           cbar_ax=cbar_ax, k=k,
                                           alpha=thealpha,
                                           minfrac=minfrac,
                                           maxfrac=maxfrac)
            #axs.flat[idx].legend(loc=legendstring,
            #                     prop={'size': legendsize},
            #                     bbox_to_anchor=legendoffset)
        

        
        xT = np.arange(0, 2*np.pi,np.pi/2)
        xL=['0',r'$\pi/2$', r'$\pi$',r'$3\pi/2}$']
        for i in np.flip(np.arange(matrix.shape[1])):
            
            fig.axes[i].set_xticks(xT)
            fig.axes[i].set_xticklabels(xL)
            #fig.axes[i].set_rgrids(yT,yL)
            fig.axes[i].yaxis.set_tick_params(labelbottom=True)
                
            fig.axes[i].set_rlabel_position(rlabelpos)
            fig.axes[i].tick_params(labelsize=labelsize)
            #print("adding labels",xL)
        if returnmatrix:
            return fig, axs, matrix
        return fig, axs