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

#ifdef FIX_CLASS

FixStyle(nve/stadium,FixNVEStadium)

#else

#ifndef LMP_FIX_NVE_STADIUM_H
#define LMP_FIX_NVE_STADIUM_H

#include "fix.h"

namespace LAMMPS_NS {

class FixNVEStadium : public Fix {
 public:
  FixNVEStadium(class LAMMPS *, int, char **);
  virtual ~FixNVEStadium();
  int setmask();
  void init();
  void initial_integrate(int);
  void final_integrate();
  void initial_integrate_respa(int, int, int);
  void final_integrate_respa(int, int);
  void reset_dt();
  double compute_scalar();
  void compute_target();

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

    /* New local function to enable stadium langevin fixes
   * Determines the 
   * @param requires 4 vales xmin, xmax, ymin and s_ymax
   * or @param require 6 values xmin, xmax, ymin, ymax, zmin and zmax
   */
  double minvalue(double, double, double, double);
  
  double minvalue(double, double, double, double, double, double);

protected:
  double *gfactor1, *gfactor2, *ratio;
  
  double t_start, t_stop, t_period, t_target;
  char *tstr;
  int tstyle,tvar;
  double tsqrt;

  int maxatom1, maxatom2;
  double *tforce;

  char *id_temp;
  class Compute *temperature;

  int nvalues;
  
  int nlevels_respa;
  int nrestart;
  class RanMars *random;
  int seed;

  /// Stadium Langevin parameters
    int stadflag; /// Stadium langevan flag
  double s_xmin, s_xmax, s_ymin, s_ymax, s_zmin, s_zmax, s_width; /// Stadium langevin parameters
  double **flangevin; // flangevin[:][0] holds the stadium langevin co-efficient, cols [1]-[3] hold the random + drag force
   
 private:
  double dtv,dtf;
  double *step_respa;
  int ncount;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

W: Should not use fix nve/limit with fix shake or fix rattle

This will lead to invalid constraint forces in the SHAKE/RATTLE
computation.

*/
