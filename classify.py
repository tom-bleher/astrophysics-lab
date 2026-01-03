def calc_galaxy(B, dB, I, dI, U, dU, V, dV, plot=False):
    galaxies=["elliptical", "S0", "Sa", "Sb", "sbt1", "sbt2", "sbt3", "sbt4", "sbt5", "sbt6"]
    chimins=[]
    zmins=[]
    for galaxy_type in galaxies:
        chi_z=dict()
        for i in numpy.arange (0,1.0,0.05):
            redshift = i
            ##################################################
    
            # load the relevant galaxy
            path = f"{parent_path}/Spectra/{galaxy_type}.dat"
            wl, spec = numpy.loadtxt(path, usecols=[0,1], unpack=True)
    
            # apply a redshift transformation to the galaxy spectrum
            wl_redshifted = wl * (1 + redshift)
    
            # move to a common wavelength grid
            wl_small = numpy.arange(2200, 9500, 1)
            spec_small = numpy.interp(wl_small, wl_redshifted, spec)
            spec_small_norm = spec_small / numpy.median(spec_small)
    
            # plot the galaxy after the redshift transformation
            #plt.title("galaxy type: %s" % galaxy_type)
            #plt.step(wl_small, spec_small_norm, "k")
            #plt.xlabel("wavelength (A)")
            #plt.ylabel("normalised flux ()")
    
            # the filters do not move, so we can just plot them as they were
            max_value = numpy.max(spec_small_norm) * 0.5
            #for i in range(len(filters)):
                #plt.step(wl_filter_small, filters[i] * max_value, color=colors_filters[i], alpha=0.5)
                #plt.fill_between(wl_filter_small, 0, filters[i] * max_value, color=colors_filters[i], alpha=0.3)
    
    
            # create the synthetic photometry according to this redshift
            syn_photometry = []
            for j in range(len(filters)):
                filter_arr = filters[j]
                syn_phot = numpy.median(filter_arr[filter_arr != 0] * spec_small_norm[filter_arr != 0])
                syn_photometry.append(syn_phot)
            syn_photometry = numpy.array(syn_photometry)
            syn_photometry_norm = syn_photometry / numpy.median(syn_photometry)
            #plt.plot(filter_centers, syn_photometry_norm, "o", color="orange", markersize=10, label="model photometry")
    
    
            # plot the measured photometry
            meas_photo = numpy.array([U, B, V, I])
            meas_errs = numpy.array([dU, dB, dV, dI])
    
            meas_photo_norm = meas_photo / numpy.median(meas_photo)
            meas_photo_err_norm = meas_errs / numpy.median(meas_photo)
            #plt.errorbar(filter_centers, meas_photo_norm, yerr=meas_photo_err_norm, fmt="+c", label="measured photometry")
    
    
            # compute the chi-square of the fit
            residuals_w = numpy.round((meas_photo_norm - syn_photometry_norm)**2 / meas_photo_err_norm**2, 2)
            residuals_st = "[%s, %s, %s, %s]" % (residuals_w[0], residuals_w[1], residuals_w[2], residuals_w[3])
            chi_square = numpy.sum((meas_photo_norm - syn_photometry_norm)**2 / meas_photo_err_norm**2)
            #plt.title("galaxy type: %s, $\\chi^2 = %s, %s $" % (galaxy_type, numpy.round(chi_square, 2), residuals_st))
    
            #plt.legend(loc="best")
    
            #plt.ylim(0, max_value * 2 * 1.5)
            chi_z[redshift]=chi_square
        #print "z-min=",min(chi_z, key=chi_z.get), "Chi-min=", chi_z[min(chi_z, key=chi_z.get)]
        best_z = builtins.min(chi_z, key=chi_z.get)
        zmins.append(best_z)
        chimins.append(chi_z[best_z])   
        
    galaxy_type=galaxies[chimins.index(min(chimins))]
    redshift = zmins[chimins.index(min(chimins))]