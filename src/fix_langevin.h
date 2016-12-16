/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

// Stadium langevin not implemented for spherical and ellipsoidal particles
// Also not implemented for rotational degrees of freedom
// Use input file with care

#ifdef FIX_CLASS

FixStyle(langevin,FixLangevin)

#else

#ifndef LMP_FIX_LANGEVIN_H
#define LMP_FIX_LANGEVIN_H

#include "fix.h"

namespace LAMMPS_NS {

class FixLangevin : public Fix {
 public:
  FixLangevin(class LAMMPS *, int, char **);
  virtual ~FixLangevin();

  int setmask();
  void init();
  void setup(int);
  virtual void post_force(int);
  void post_force_respa(int, int, int);
  virtual void end_of_step();
  
  int modify_param(int, char **);
  virtual double compute_scalar();
  
  /* Memomry Methods
   */
  double memory_usage();
  virtual void *extract(const char *, int &);
  void grow_arrays(int);
  void copy_arrays(int, int, int);
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);

  /* Restart methods 
   */
  /*
  void write_restart(FILE *);
  void restart(char *);
  */
  
  int pack_restart(int, double *);
  void unpack_restart(int, int);
  int maxsize_restart();
  int size_restart(int);
  
  void reset_target(double);
  void reset_dt();
  
  /* New local function to enable stadium langevin fixes
   * Determines the 
   * @param requires 4 vales xmin, xmax, ymin and s_ymax
   * or @param require 6 values xmin, xmax, ymin, ymax, zmin and zmax
   */
  double minvalue(double, double, double, double);
  
  double minvalue(double, double, double, double, double, double);

 protected:
  int gjfflag,oflag,tallyflag,zeroflag,tbiasflag;
  
  double ascale;
  double t_start,t_stop,t_period,t_target;
  
  
  double *gfactor1,*gfactor2,*ratio;
  double energy,energy_onestep;
  double tsqrt;
  int tstyle,tvar;
  double gjffac;
  char *tstr;

  class AtomVecEllipsoid *avec;

  int maxatom1,maxatom2;
  double **flangevin;
  double *tforce;
  double **franprev;
  
  
  double current_time;
  
  
  int nvalues;

  char *id_temp;
  class Compute *temperature;

  int nlevels_respa;
  int nrestart;
  class RanMars *random;
  int seed;

  /// Stadium Langevin parameters
    int stadflag; /// Stadium langevan flag
  double s_xmin, s_xmax, s_ymin, s_ymax, s_zmin, s_zmax, s_width; /// Stadium langevin parameters
  double *gamma_stadium; /// Stadium damping coefficients one per atom
  
  
  // comment next line to turn off templating
//#define TEMPLATED_FIX_LANGEVIN
#ifdef TEMPLATED_FIX_LANGEVIN
  template < int Tp_TSTYLEATOM, int Tp_GJF, int Tp_TALLY,
	     int Tp_BIAS, int Tp_RMASS, int Tp_ZERO >
  void post_force_templated();
#else
  void post_force_untemplated(int, int, int,
			      int, int, int, int);
#endif
  void omega_thermostat();
  void angmom_thermostat();
  void compute_target();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Fix langevin period must be > 0.0

The time window for temperature relaxation must be > 0

W: Energy tally does not account for 'zero yes'

The energy removed by using the 'zero yes' flag is not accounted
for in the energy tally and thus energy conservation cannot be
monitored in this case.

E: Fix langevin omega requires atom style sphere

Self-explanatory.

E: Fix langevin angmom requires atom style ellipsoid

Self-explanatory.

E: Variable name for fix langevin does not exist

Self-explanatory.

E: Variable for fix langevin is invalid style

It must be an equal-style variable.

E: Fix langevin omega requires extended particles

One of the particles has radius 0.0.

E: Fix langevin angmom requires extended particles

This fix option cannot be used with point paritlces.

E: Cannot zero Langevin force of 0 atoms

The group has zero atoms, so you cannot request its force
be zeroed.

E: Fix langevin variable returned negative temperature

Self-explanatory.

E: Could not find fix_modify temperature ID

The compute ID for computing temperature does not exist.

E: Fix_modify temperature ID does not compute temperature

The compute ID assigned to the fix must compute temperature.

W: Group for fix_modify temp != fix group

The fix_modify command is specifying a temperature computation that
computes a temperature on a different group of atoms than the fix
itself operates on.  This is probably not what you want to do.



*/
