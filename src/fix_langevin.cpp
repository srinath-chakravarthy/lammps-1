/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Carolyn Phillips (U Mich), reservoir energy tally
                         Aidan Thompson (SNL) GJF formulation
------------------------------------------------------------------------- */

#include <mpi.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "fix_langevin.h"
#include "math_extra.h"
#include "atom.h"
#include "atom_vec_ellipsoid.h"
#include "force.h"
#include "update.h"
#include "modify.h"
#include "compute.h"
#include "domain.h"
#include "region.h"
#include "respa.h"
#include "comm.h"
#include "input.h"
#include "variable.h"
#include "random_mars.h"
#include "memory.h"
#include "error.h"
#include "group.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{NOBIAS,BIAS};
enum{CONSTANT,EQUAL,ATOM};

#define SINERTIA 0.4          // moment of inertia prefactor for sphere
#define EINERTIA 0.2          // moment of inertia prefactor for ellipsoid

/* ---------------------------------------------------------------------- */

FixLangevin::FixLangevin(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  gjfflag(0), gfactor1(NULL), gfactor2(NULL), ratio(NULL), tstr(NULL),
  flangevin(NULL), tforce(NULL), franprev(NULL), id_temp(NULL), random(NULL)
{
  if (narg < 7) error->all(FLERR,"Illegal fix langevin command");

  dynamic_group_allow = 1;
  scalar_flag = 1;
  global_freq = 1;
  extscalar = 1;
  nevery = 1;

  
  /* Flags to output per atom quantities 
   */
  peratom_flag = 1; /// Sets per atom quantity output
  peratom_freq = 1; /// Set per atom output frequency 
  size_peratom_cols = 0; /// Sets the size of per atom arrays/vectors
  
  /* Flags for restart
   */ 
  restart_peratom = 1; /// Save per atom state to restart file
  
  tstr = NULL;
  if (strstr(arg[3],"v_") == arg[3]) {
    int n = strlen(&arg[3][2]) + 1;
    tstr = new char[n];
    strcpy(tstr,&arg[3][2]);
  } else {
    t_start = force->numeric(FLERR,arg[3]);
    t_target = t_start;
    tstyle = CONSTANT;
  }

  t_stop = force->numeric(FLERR,arg[4]);
  t_period = force->numeric(FLERR,arg[5]);
  seed = force->inumeric(FLERR,arg[6]);

  if (t_period <= 0.0) error->all(FLERR,"Fix langevin period must be > 0.0");
  if (seed <= 0) error->all(FLERR,"Illegal fix langevin command");

  // initialize Marsaglia RNG with processor-unique seed

  random = new RanMars(lmp,seed + comm->me);

  // allocate per-type arrays for force prefactors

  gfactor1 = new double[atom->ntypes+1];
  gfactor2 = new double[atom->ntypes+1];
  ratio = new double[atom->ntypes+1];

  // optional args

  for (int i = 1; i <= atom->ntypes; i++) ratio[i] = 1.0;
  ascale = 0.0;
  gjfflag = 0;
  oflag = 0;
  tallyflag = 0;
  zeroflag = 0;
  stadflag = 0;
  
  int iarg = 7;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"angmom") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix langevin command");
      if (strcmp(arg[iarg+1],"no") == 0) ascale = 0.0;
      else ascale = force->numeric(FLERR,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"gjf") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix langevin command");
      if (strcmp(arg[iarg+1],"no") == 0) gjfflag = 0;
      else if (strcmp(arg[iarg+1],"yes") == 0) gjfflag = 1;
      else error->all(FLERR,"Illegal fix langevin command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"omega") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix langevin command");
      if (strcmp(arg[iarg+1],"no") == 0) oflag = 0;
      else if (strcmp(arg[iarg+1],"yes") == 0) oflag = 1;
      else error->all(FLERR,"Illegal fix langevin command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"scale") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix langevin command");
      int itype = force->inumeric(FLERR,arg[iarg+1]);
      double scale = force->numeric(FLERR,arg[iarg+2]);
      if (itype <= 0 || itype > atom->ntypes)
        error->all(FLERR,"Illegal fix langevin command");
      ratio[itype] = scale;
      iarg += 3;
    } else if (strcmp(arg[iarg],"tally") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix langevin command");
      if (strcmp(arg[iarg+1],"no") == 0) tallyflag = 0;
      else if (strcmp(arg[iarg+1],"yes") == 0) tallyflag = 1;
      else error->all(FLERR,"Illegal fix langevin command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"zero") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix langevin command");
      if (strcmp(arg[iarg+1],"no") == 0) zeroflag = 0;
      else if (strcmp(arg[iarg+1],"yes") == 0) zeroflag = 1;
      else error->all(FLERR,"Illegal fix langevin command");
      iarg += 2;
/*  Stadium Langevin part of thermostat 
    Check if flag is set to yes or no
    if no, then do not check for anything else
      if yes -> then check to make sure other arguments exist and need to exist
      these are xmin, xmax, ymin, ymax and width
*/
    } else if (strcmp(arg[iarg],"stadium") == 0) { 
      if (iarg+8 > narg) error->all(FLERR,"Illegal fix langevin command (stadium)");
      s_xmin = force->numeric(FLERR,arg[iarg+1]);
      s_xmax = force->numeric(FLERR,arg[iarg+2]);
      s_ymin = force->numeric(FLERR,arg[iarg+3]);
      s_ymax = force->numeric(FLERR,arg[iarg+4]);
      s_zmin = force->numeric(FLERR,arg[iarg+5]);
      s_zmax = force->numeric(FLERR,arg[iarg+6]);
      s_width = force->numeric(FLERR,arg[iarg+7]);
      if (s_xmin > s_xmax) error->all(FLERR, "Illegal stadium langevin, xmax must be greater than xmin");
      if (s_ymin > s_ymax) error->all(FLERR, "Illegal stadium langevin, ymax must be greater than ymin");
      if (s_zmin > s_zmax) error->all(FLERR, "Illegal stadium langevin, zmax must be greater than zmin");
      if (s_width < 0.0) error->all(FLERR, "Illegal stadium langevin, width must be greater than 0.0");
      stadflag = 1;
      iarg += 8;
    } else error->all(FLERR,"Illegal fix langevin command");
  }

  // set temperature = NULL, user can override via fix_modify if wants bias

  id_temp = NULL;
  temperature = NULL;

  // flangevin is unallocated until first call to setup()
  // compute_scalar checks for this and returns 0.0 if flangevin is NULL

  energy = 0.0;
  flangevin = NULL;
  franprev = NULL;
  tforce = NULL;
  maxatom1 = maxatom2 = 0;

  // setup atom-based array for franprev
  // register with Atom class
  // no need to set peratom_flag, b/c data is for internal use only

  if (gjfflag) {
    nvalues = 3;
    grow_arrays(atom->nmax);
    atom->add_callback(0);

  // initialize franprev to zero

    int nlocal = atom->nlocal;
    for (int i = 0; i < nlocal; i++) {
      franprev[i][0] = 0.0;
      franprev[i][1] = 0.0;
      franprev[i][2] = 0.0;
    }
  }
  // Setup atom-based array gamma_stadium 
  // Allocate space and initialize to zero 
  // grow_arrays method modified to include extra parameter integer 
  //    that allows for the possibility of choosing either gamma_stadium or franprev
  // Not sure of what atom_callback does, but it is called again here 
  //   perhaps it should be instantiated only once of not already
  gamma_stadium = NULL;
  if (stadflag){
    nvalues = 3;
    grow_arrays(atom->nmax);
    atom->add_callback(0);  /// Call back for new to grow arrays
    atom->add_callback(1);  /// Call back for restart 
    
    // Initialize the stadium langevin peratom damping coefficient factor
    // 	values are issued as scale factors which are equal to 1.0 - abs(min(x-xmin, x-xmax, y-ymin, y-max)
    int nlocal = atom->nlocal;
    for (int i = 0; i < nlocal; i++){
      gamma_stadium[i] = 0.0;
    }
      // Stadium langevin initiliazation
      if (stadflag){
	int nlocal = atom->nlocal;
	double **x = atom->x;
	int *mask = atom->mask;
	for (int i = 0; i < nlocal; i++){
	  if (mask[i] & groupbit) {
	    double x1 = fabs(x[i][0] - s_xmin);
	    double x2 = fabs(x[i][0] - s_xmax);
	    double y1 = fabs(x[i][1] - s_ymin);
	    double y2 = fabs(x[i][1] - s_ymax);
	    double z1 = fabs(x[i][2] - s_zmin);
	    double z2 = fabs(x[i][2] - s_zmax);

	    double dxy = (minvalue(x1, x2, y1, y2, z1, z2));
	    gamma_stadium[i] = 1.0 - dxy/s_width;
	  } else
	  {
	    gamma_stadium[i] = 0.0;
	  }
	}
      }
  }
  //nrestart = size of per-atom restart data;
  //nrestart = 1 + gamma_stadium;
  nrestart = 1;
  if (stadflag) nrestart++;
  
  if (tallyflag && zeroflag && comm->me == 0)
    error->warning(FLERR,"Energy tally does not account for 'zero yes'");
}

/* ---------------------------------------------------------------------- */

FixLangevin::~FixLangevin()
{
  delete random;
  delete [] tstr;
  delete [] gfactor1;
  delete [] gfactor2;
  delete [] ratio;
  delete [] id_temp;
  memory->destroy(flangevin);
  memory->destroy(tforce);

  if (gjfflag) {
    memory->destroy(franprev);
    atom->delete_callback(id,0);
  }
  if (stadflag){
    memory->destroy(gamma_stadium);
    atom->delete_callback(id,0);
    atom->delete_callback(id,1);
  }
}

/* ---------------------------------------------------------------------- */

int FixLangevin::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= END_OF_STEP;
  mask |= THERMO_ENERGY;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixLangevin::init()
{
  if (oflag && !atom->sphere_flag)
    error->all(FLERR,"Fix langevin omega requires atom style sphere");
  if (ascale && !atom->ellipsoid_flag)
    error->all(FLERR,"Fix langevin angmom requires atom style ellipsoid");

  // check variable

  if (tstr) {
    tvar = input->variable->find(tstr);
    if (tvar < 0)
      error->all(FLERR,"Variable name for fix langevin does not exist");
    if (input->variable->equalstyle(tvar)) tstyle = EQUAL;
    else if (input->variable->atomstyle(tvar)) tstyle = ATOM;
    else error->all(FLERR,"Variable for fix langevin is invalid style");
  }

  // if oflag or ascale set, check that all group particles are finite-size

  if (oflag) {
    double *radius = atom->radius;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit)
        if (radius[i] == 0.0)
          error->one(FLERR,"Fix langevin omega requires extended particles");
  }

  if (ascale) {
    avec = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
    if (!avec)
      error->all(FLERR,"Fix langevin angmom requires atom style ellipsoid");

    int *ellipsoid = atom->ellipsoid;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit)
        if (ellipsoid[i] < 0)
          error->one(FLERR,"Fix langevin angmom requires extended particles");
  }

  // set force prefactors

  if (!atom->rmass) {
    for (int i = 1; i <= atom->ntypes; i++) {
      gfactor1[i] = -atom->mass[i] / t_period / force->ftm2v;
      gfactor2[i] = sqrt(atom->mass[i]) *
        sqrt(24.0*force->boltz/t_period/update->dt/force->mvv2e) /
        force->ftm2v;
      gfactor1[i] *= 1.0/ratio[i];
      gfactor2[i] *= 1.0/sqrt(ratio[i]);
    }
  }

  if (temperature && temperature->tempbias) tbiasflag = BIAS;
  else tbiasflag = NOBIAS;

  if (strstr(update->integrate_style,"respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;

  if (gjfflag) gjffac = 1.0/(1.0+update->dt/2.0/t_period);
  
}

/* ---------------------------------------------------------------------- */

void FixLangevin::setup(int vflag)
{
  if (strstr(update->integrate_style,"verlet"))
    post_force(vflag);
  else {
    ((Respa *) update->integrate)->copy_flevel_f(nlevels_respa-1);
    post_force_respa(vflag,nlevels_respa-1,0);
    ((Respa *) update->integrate)->copy_f_flevel(nlevels_respa-1);
  }
}

/* ---------------------------------------------------------------------- */

void FixLangevin::post_force(int vflag)
{
  double *rmass = atom->rmass;

  // enumerate all 2^6 possibilities for template parameters
  // this avoids testing them inside inner loop:
  // TSTYLEATOM, GJF, TALLY, BIAS, RMASS, ZERO

#ifdef TEMPLATED_FIX_LANGEVIN
  if (tstyle == ATOM)
    if (gjfflag)
      if (tallyflag)
	if (tbiasflag == BIAS)
	  if (rmass)
	    if (zeroflag) post_force_templated<1,1,1,1,1,1>();
            else          post_force_templated<1,1,1,1,1,0>();
	  else
	    if (zeroflag) post_force_templated<1,1,1,1,0,1>();
            else          post_force_templated<1,1,1,1,0,0>();
	else
	  if (rmass)
	    if (zeroflag) post_force_templated<1,1,1,0,1,1>();
	    else          post_force_templated<1,1,1,0,1,0>();
	  else
	    if (zeroflag) post_force_templated<1,1,1,0,0,1>();
	    else          post_force_templated<1,1,1,0,0,0>();
      else
	if (tbiasflag == BIAS)
	  if (rmass)
	    if (zeroflag) post_force_templated<1,1,0,1,1,1>();
	    else          post_force_templated<1,1,0,1,1,0>();
	  else
	    if (zeroflag) post_force_templated<1,1,0,1,0,1>();
	    else          post_force_templated<1,1,0,1,0,0>();
	else
	  if (rmass)
	    if (zeroflag) post_force_templated<1,1,0,0,1,1>();
	    else          post_force_templated<1,1,0,0,1,0>();
	  else
	    if (zeroflag) post_force_templated<1,1,0,0,0,1>();
	    else          post_force_templated<1,1,0,0,0,0>();
    else
      if (tallyflag)
	if (tbiasflag == BIAS)
	  if (rmass)
	    if (zeroflag) post_force_templated<1,0,1,1,1,1>();
	    else          post_force_templated<1,0,1,1,1,0>();
	  else
	    if (zeroflag) post_force_templated<1,0,1,1,0,1>();
	    else          post_force_templated<1,0,1,1,0,0>();
	else
	  if (rmass)
	    if (zeroflag) post_force_templated<1,0,1,0,1,1>();
	    else          post_force_templated<1,0,1,0,1,0>();
	  else
	    if (zeroflag) post_force_templated<1,0,1,0,0,1>();
	    else          post_force_templated<1,0,1,0,0,0>();
      else
	if (tbiasflag == BIAS)
	  if (rmass)
	    if (zeroflag) post_force_templated<1,0,0,1,1,1>();
	    else          post_force_templated<1,0,0,1,1,0>();
	  else
	    if (zeroflag) post_force_templated<1,0,0,1,0,1>();
	    else          post_force_templated<1,0,0,1,0,0>();
	else
	  if (rmass)
	    if (zeroflag) post_force_templated<1,0,0,0,1,1>();
	    else          post_force_templated<1,0,0,0,1,0>();
	  else
	    if (zeroflag) post_force_templated<1,0,0,0,0,1>();
	    else          post_force_templated<1,0,0,0,0,0>();
  else
    if (gjfflag)
      if (tallyflag)
	if (tbiasflag == BIAS)
	  if (rmass)
	    if (zeroflag) post_force_templated<0,1,1,1,1,1>();
	    else          post_force_templated<0,1,1,1,1,0>();
	  else
	    if (zeroflag) post_force_templated<0,1,1,1,0,1>();
	    else          post_force_templated<0,1,1,1,0,0>();
	else
	  if (rmass)
	    if (zeroflag) post_force_templated<0,1,1,0,1,1>();
	    else          post_force_templated<0,1,1,0,1,0>();
	  else
	    if (zeroflag) post_force_templated<0,1,1,0,0,1>();
	    else          post_force_templated<0,1,1,0,0,0>();
      else
	if (tbiasflag == BIAS)
	  if (rmass)
	    if (zeroflag) post_force_templated<0,1,0,1,1,1>();
	    else          post_force_templated<0,1,0,1,1,0>();
	  else
	    if (zeroflag) post_force_templated<0,1,0,1,0,1>();
	    else          post_force_templated<0,1,0,1,0,0>();
	else
	  if (rmass)
	    if (zeroflag) post_force_templated<0,1,0,0,1,1>();
	    else          post_force_templated<0,1,0,0,1,0>();
	  else
	    if (zeroflag) post_force_templated<0,1,0,0,0,1>();
	    else          post_force_templated<0,1,0,0,0,0>();
    else
      if (tallyflag)
	if (tbiasflag == BIAS)
	  if (rmass)
	    if (zeroflag) post_force_templated<0,0,1,1,1,1>();
	    else          post_force_templated<0,0,1,1,1,0>();
	  else
	    if (zeroflag) post_force_templated<0,0,1,1,0,1>();
	    else          post_force_templated<0,0,1,1,0,0>();
	else
	  if (rmass)
	    if (zeroflag) post_force_templated<0,0,1,0,1,1>();
	    else          post_force_templated<0,0,1,0,1,0>();
	  else
	    if (zeroflag) post_force_templated<0,0,1,0,0,1>();
	    else          post_force_templated<0,0,1,0,0,0>();
      else
	if (tbiasflag == BIAS)
	  if (rmass)
	    if (zeroflag) post_force_templated<0,0,0,1,1,1>();
	    else          post_force_templated<0,0,0,1,1,0>();
	  else
	    if (zeroflag) post_force_templated<0,0,0,1,0,1>();
	    else          post_force_templated<0,0,0,1,0,0>();
	else
	  if (rmass)
	    if (zeroflag) post_force_templated<0,0,0,0,1,1>();
	    else          post_force_templated<0,0,0,0,1,0>();
	  else
	    if (zeroflag) post_force_templated<0,0,0,0,0,1>();
	    else          post_force_templated<0,0,0,0,0,0>();
#else
  post_force_untemplated(int(tstyle==ATOM), gjfflag, tallyflag,
			 int(tbiasflag==BIAS), int(rmass!=NULL), zeroflag, stadflag);
#endif
}

/* ---------------------------------------------------------------------- */

void FixLangevin::post_force_respa(int vflag, int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_force(vflag);
}

/* ----------------------------------------------------------------------
   modify forces using one of the many Langevin styles
------------------------------------------------------------------------- */

#ifdef TEMPLATED_FIX_LANGEVIN
template < int Tp_TSTYLEATOM, int Tp_GJF, int Tp_TALLY,
	   int Tp_BIAS, int Tp_RMASS, int Tp_ZERO >
void FixLangevin::post_force_templated()
#else
void FixLangevin::post_force_untemplated
  (int Tp_TSTYLEATOM, int Tp_GJF, int Tp_TALLY,
   int Tp_BIAS, int Tp_RMASS, int Tp_ZERO, int Tp_stadflag)
#endif
{
  double gamma1,gamma2;

  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  // apply damping and thermostat to atoms in group

  // for Tp_TSTYLEATOM:
  //   use per-atom per-coord target temperature
  // for Tp_GJF:
  //   use Gronbech-Jensen/Farago algorithm
  //   else use regular algorithm
  // for Tp_TALLY:
  //   store drag plus random forces in flangevin[nlocal][3]
  // for Tp_BIAS:
  //   calculate temperature since some computes require temp
  //   computed on current nlocal atoms to remove bias
  //   test v = 0 since some computes mask non-participating atoms via v = 0
  //   and added force has extra term not multiplied by v = 0
  // for Tp_RMASS:
  //   use per-atom masses
  //   else use per-type masses
  // for Tp_ZERO:
  //   sum random force over all atoms in group
  //   subtract sum/count from each atom in group

  double fdrag[3],fran[3],fsum[3],fsumall[3];
  bigint count;
  double fswap;

  double boltz = force->boltz;
  double dt = update->dt;
  double mvv2e = force->mvv2e;
  double ftm2v = force->ftm2v;

  compute_target();

  if (Tp_ZERO) {
    fsum[0] = fsum[1] = fsum[2] = 0.0;
    count = group->count(igroup);
    if (count == 0)
      error->all(FLERR,"Cannot zero Langevin force of 0 atoms");
  }

  // reallocate flangevin if necessary

  if (Tp_TALLY) {
    if (atom->nmax > maxatom1) {
      memory->destroy(flangevin);
      maxatom1 = atom->nmax;
      memory->create(flangevin,maxatom1,3,"langevin:flangevin");
    }
  }

  if (Tp_BIAS) temperature->compute_scalar();

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      if (Tp_TSTYLEATOM) tsqrt = sqrt(tforce[i]);
      if (Tp_RMASS) {
	gamma1 = -rmass[i] / t_period / ftm2v;
	gamma2 = sqrt(rmass[i]) * sqrt(24.0*boltz/t_period/dt/mvv2e) / ftm2v;
	gamma1 *= 1.0/ratio[type[i]];
	gamma2 *= 1.0/sqrt(ratio[type[i]]) * tsqrt;
	
	// Add stadium langevin to gamma factors 
	if (Tp_stadflag){
	  gamma1 *= gamma_stadium[i];
	  gamma2 *= sqrt(gamma_stadium[i]);
	}
      } else {
	gamma1 = gfactor1[type[i]];
	gamma2 = gfactor2[type[i]] * tsqrt;

	// Add stadium langevin to gamma factors 
	if (Tp_stadflag){
	  gamma1 *= gamma_stadium[i];
	  gamma2 *= sqrt(gamma_stadium[i]);
	  if (!(isfinite(gamma1)) || !(isfinite(gamma2))){
	    error->one(FLERR,"Fix langevin factors are infinite");
	  }
	}

	
      }

      fran[0] = gamma2*(random->uniform()-0.5);
      fran[1] = gamma2*(random->uniform()-0.5);
      fran[2] = gamma2*(random->uniform()-0.5);

      if (Tp_BIAS) {
	temperature->remove_bias(i,v[i]);
	fdrag[0] = gamma1*v[i][0];
	fdrag[1] = gamma1*v[i][1];
	fdrag[2] = gamma1*v[i][2];
	if (v[i][0] == 0.0) fran[0] = 0.0;
	if (v[i][1] == 0.0) fran[1] = 0.0;
	if (v[i][2] == 0.0) fran[2] = 0.0;
	temperature->restore_bias(i,v[i]);
      } else {
	fdrag[0] = gamma1*v[i][0];
	fdrag[1] = gamma1*v[i][1];
	fdrag[2] = gamma1*v[i][2];
      }

      if (Tp_GJF) {
	fswap = 0.5*(fran[0]+franprev[i][0]);
	franprev[i][0] = fran[0];
	fran[0] = fswap;
	fswap = 0.5*(fran[1]+franprev[i][1]);
	franprev[i][1] = fran[1];
	fran[1] = fswap;
	fswap = 0.5*(fran[2]+franprev[i][2]);
	franprev[i][2] = fran[2];
	fran[2] = fswap;

	fdrag[0] *= gjffac;
	fdrag[1] *= gjffac;
	fdrag[2] *= gjffac;
	fran[0] *= gjffac;
	fran[1] *= gjffac;
	fran[2] *= gjffac;
	f[i][0] *= gjffac;
	f[i][1] *= gjffac;
	f[i][2] *= gjffac;
      }

      f[i][0] += fdrag[0] + fran[0];
      f[i][1] += fdrag[1] + fran[1];
      f[i][2] += fdrag[2] + fran[2];

      if (Tp_TALLY) {
	flangevin[i][0] = fdrag[0] + fran[0];
	flangevin[i][1] = fdrag[1] + fran[1];
	flangevin[i][2] = fdrag[2] + fran[2];
      }

      if (Tp_ZERO) {
	fsum[0] += fran[0];
	fsum[1] += fran[1];
	fsum[2] += fran[2];
      }
    }
  }

  // set total force to zero

  if (Tp_ZERO) {
    MPI_Allreduce(fsum,fsumall,3,MPI_DOUBLE,MPI_SUM,world);
    fsumall[0] /= count;
    fsumall[1] /= count;
    fsumall[2] /= count;
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        f[i][0] -= fsumall[0];
        f[i][1] -= fsumall[1];
        f[i][2] -= fsumall[2];
      }
    }
  }

  // thermostat omega and angmom

  if (oflag) omega_thermostat();
  if (ascale) angmom_thermostat();
}

/* ----------------------------------------------------------------------
   set current t_target and t_sqrt
------------------------------------------------------------------------- */

void FixLangevin::compute_target()
{
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double delta = update->ntimestep - update->beginstep;
  if (delta != 0.0) delta /= update->endstep - update->beginstep;

  // if variable temp, evaluate variable, wrap with clear/add
  // reallocate tforce array if necessary

  if (tstyle == CONSTANT) {
    t_target = t_start + delta * (t_stop-t_start);
    tsqrt = sqrt(t_target);
  } else {
    modify->clearstep_compute();
    if (tstyle == EQUAL) {
      t_target = input->variable->compute_equal(tvar);
      if (t_target < 0.0)
        error->one(FLERR,"Fix langevin variable returned negative temperature");
      tsqrt = sqrt(t_target);
    } else {
      if (atom->nmax > maxatom2) {
        maxatom2 = atom->nmax;
        memory->destroy(tforce);
        memory->create(tforce,maxatom2,"langevin:tforce");
      }
      input->variable->compute_atom(tvar,igroup,tforce,1,0);
      for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit)
            if (tforce[i] < 0.0)
              error->one(FLERR,
                         "Fix langevin variable returned negative temperature");
    }
    modify->addstep_compute(update->ntimestep + 1);
  }
}

/* ----------------------------------------------------------------------
   thermostat rotational dof via omega
------------------------------------------------------------------------- */

void FixLangevin::omega_thermostat()
{
  double gamma1,gamma2;

  double boltz = force->boltz;
  double dt = update->dt;
  double mvv2e = force->mvv2e;
  double ftm2v = force->ftm2v;

  double **torque = atom->torque;
  double **omega = atom->omega;
  double *radius = atom->radius;
  double *rmass = atom->rmass;
  int *mask = atom->mask;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  // rescale gamma1/gamma2 by 10/3 & sqrt(10/3) for spherical particles
  // does not affect rotational thermosatting
  // gives correct rotational diffusivity behavior

  double tendivthree = 10.0/3.0;
  double tran[3];
  double inertiaone;

  for (int i = 0; i < nlocal; i++) {
    if ((mask[i] & groupbit) && (radius[i] > 0.0)) {
      inertiaone = SINERTIA*radius[i]*radius[i]*rmass[i];
      if (tstyle == ATOM) tsqrt = sqrt(tforce[i]);
      gamma1 = -tendivthree*inertiaone / t_period / ftm2v;
      gamma2 = sqrt(inertiaone) * sqrt(80.0*boltz/t_period/dt/mvv2e) / ftm2v;
      gamma1 *= 1.0/ratio[type[i]];
      gamma2 *= 1.0/sqrt(ratio[type[i]]) * tsqrt;
      tran[0] = gamma2*(random->uniform()-0.5);
      tran[1] = gamma2*(random->uniform()-0.5);
      tran[2] = gamma2*(random->uniform()-0.5);
      torque[i][0] += gamma1*omega[i][0] + tran[0];
      torque[i][1] += gamma1*omega[i][1] + tran[1];
      torque[i][2] += gamma1*omega[i][2] + tran[2];
    }
  }
}

/* ----------------------------------------------------------------------
   thermostat rotational dof via angmom
------------------------------------------------------------------------- */

void FixLangevin::angmom_thermostat()
{
  double gamma1,gamma2;

  double boltz = force->boltz;
  double dt = update->dt;
  double mvv2e = force->mvv2e;
  double ftm2v = force->ftm2v;

  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  double **torque = atom->torque;
  double **angmom = atom->angmom;
  double *rmass = atom->rmass;
  int *ellipsoid = atom->ellipsoid;
  int *mask = atom->mask;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  // rescale gamma1/gamma2 by ascale for aspherical particles
  // does not affect rotational thermosatting
  // gives correct rotational diffusivity behavior if (nearly) spherical
  // any value will be incorrect for rotational diffusivity if aspherical

  double inertia[3],omega[3],tran[3];
  double *shape,*quat;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      shape = bonus[ellipsoid[i]].shape;
      inertia[0] = EINERTIA*rmass[i] * (shape[1]*shape[1]+shape[2]*shape[2]);
      inertia[1] = EINERTIA*rmass[i] * (shape[0]*shape[0]+shape[2]*shape[2]);
      inertia[2] = EINERTIA*rmass[i] * (shape[0]*shape[0]+shape[1]*shape[1]);
      quat = bonus[ellipsoid[i]].quat;
      MathExtra::mq_to_omega(angmom[i],quat,inertia,omega);

      if (tstyle == ATOM) tsqrt = sqrt(tforce[i]);
      gamma1 = -ascale / t_period / ftm2v;
      gamma2 = sqrt(ascale*24.0*boltz/t_period/dt/mvv2e) / ftm2v;
      gamma1 *= 1.0/ratio[type[i]];
      gamma2 *= 1.0/sqrt(ratio[type[i]]) * tsqrt;
      tran[0] = sqrt(inertia[0])*gamma2*(random->uniform()-0.5);
      tran[1] = sqrt(inertia[1])*gamma2*(random->uniform()-0.5);
      tran[2] = sqrt(inertia[2])*gamma2*(random->uniform()-0.5);
      torque[i][0] += inertia[0]*gamma1*omega[0] + tran[0];
      torque[i][1] += inertia[1]*gamma1*omega[1] + tran[1];
      torque[i][2] += inertia[2]*gamma1*omega[2] + tran[2];
    }
  }
}

/* ----------------------------------------------------------------------
   tally energy transfer to thermal reservoir
------------------------------------------------------------------------- */

void FixLangevin::end_of_step()
{
  if (!tallyflag) return;

  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  energy_onestep = 0.0;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit)
      energy_onestep += flangevin[i][0]*v[i][0] + flangevin[i][1]*v[i][1] +
        flangevin[i][2]*v[i][2];

  energy += energy_onestep*update->dt;
}

/* ---------------------------------------------------------------------- */

void FixLangevin::reset_target(double t_new)
{
  t_target = t_start = t_stop = t_new;
}

/* ---------------------------------------------------------------------- */

void FixLangevin::reset_dt()
{
  if (atom->mass) {
    for (int i = 1; i <= atom->ntypes; i++) {
      gfactor2[i] = sqrt(atom->mass[i]) *
        sqrt(24.0*force->boltz/t_period/update->dt/force->mvv2e) /
        force->ftm2v;
      gfactor2[i] *= 1.0/sqrt(ratio[i]);
    }
  }
}

/* ---------------------------------------------------------------------- */

int FixLangevin::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"temp") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal fix_modify command");
    delete [] id_temp;
    int n = strlen(arg[1]) + 1;
    id_temp = new char[n];
    strcpy(id_temp,arg[1]);

    int icompute = modify->find_compute(id_temp);
    if (icompute < 0)
      error->all(FLERR,"Could not find fix_modify temperature ID");
    temperature = modify->compute[icompute];

    if (temperature->tempflag == 0)
      error->all(FLERR,
                 "Fix_modify temperature ID does not compute temperature");
    if (temperature->igroup != igroup && comm->me == 0)
      error->warning(FLERR,"Group for fix_modify temp != fix group");
    return 2;
  }
  return 0;
}

/* ---------------------------------------------------------------------- */

double FixLangevin::compute_scalar()
{
  if (!tallyflag || flangevin == NULL) return 0.0;

  // capture the very first energy transfer to thermal reservoir

  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  if (update->ntimestep == update->beginstep) {
    energy_onestep = 0.0;
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit)
        energy_onestep += flangevin[i][0]*v[i][0] + flangevin[i][1]*v[i][1] +
          flangevin[i][2]*v[i][2];
    energy = 0.5*energy_onestep*update->dt;
  }

  // convert midstep energy back to previous fullstep energy

  double energy_me = energy - 0.5*energy_onestep*update->dt;

  double energy_all;
  MPI_Allreduce(&energy_me,&energy_all,1,MPI_DOUBLE,MPI_SUM,world);
  return -energy_all;
}

/* ----------------------------------------------------------------------
   extract thermostat properties
------------------------------------------------------------------------- */

void *FixLangevin::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str,"t_target") == 0) {
    return &t_target;
  }
  return NULL;
}

/* ----------------------------------------------------------------------
   memory usage of tally array
------------------------------------------------------------------------- */

double FixLangevin::memory_usage()
{
  double bytes = 0.0;
  if (gjfflag) bytes += atom->nmax*3 * sizeof(double);
  if (tallyflag) bytes += atom->nmax*3 * sizeof(double);
  if (tforce) bytes += atom->nmax * sizeof(double);
  // Stadium Langevin memory usage
  if (stadflag) bytes += atom->nmax * sizeof(double);
  return bytes;
}


/* ----------------------------------------------------------------------
   allocate atom-based array for franprev
------------------------------------------------------------------------- */

void FixLangevin::grow_arrays(int nmax)
{
  if (gjfflag) {
    memory->grow(franprev,nmax,3,"fix_langevin:franprev");
  } 
  if (stadflag) {
    memory->grow(gamma_stadium,nmax,"fix_langevin:gamma_stadium");
    vector_atom = gamma_stadium;
  }
}

/* ----------------------------------------------------------------------
   copy values within local atom-based array
------------------------------------------------------------------------- */

void FixLangevin::copy_arrays(int i, int j, int delflag)
{
  if (gjfflag) {
    for (int m = 0; m < nvalues; m++)
      franprev[j][m] = franprev[i][m];
  }
  if (stadflag) {
    gamma_stadium[j] = gamma_stadium[i];
  }
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixLangevin::pack_exchange(int i, double *buf)
{
  int n = 0;
  if (gjfflag) {
    for (int m = 0; m < nvalues; m++) {
      buf[n++] = franprev[i][m];
    }
    //return nvalues;
  }
  if (stadflag){
    buf[n++] = gamma_stadium[i];
  }
  return n;

}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixLangevin::unpack_exchange(int nlocal, double *buf)
{
  int n = 0;
  if (gjfflag) {
    for (int m = 0; m < nvalues; m++) {
      franprev[nlocal][m] = buf[n++];
    }
  }
  if (stadflag){
    gamma_stadium[nlocal] = buf[n++];
  }
  return n;   
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for restart file
------------------------------------------------------------------------- */

int FixLangevin::pack_restart(int i, double *buf)
{
  buf[0] = 2;
  buf[1] = gamma_stadium[i];
  return 2;
}

/* ----------------------------------------------------------------------
   unpack values from atom->extra array to restart the fix
------------------------------------------------------------------------- */

void FixLangevin::unpack_restart(int nlocal, int nth)
{
  double **extra = atom->extra;

  // skip to Nth set of extra values

  int m = 0;
  for (int i = 0; i < nth; i++) m += static_cast<int> (extra[nlocal][m]);
  m++;

  gamma_stadium[nlocal] = extra[nlocal][m++];
}


/* ----------------------------------------------------------------------
   maxsize of any atom's restart data
------------------------------------------------------------------------- */

int FixLangevin::maxsize_restart()
{
  return 2;
}

/* ----------------------------------------------------------------------
   size of atom nlocal's restart data
------------------------------------------------------------------------- */

int FixLangevin::size_restart(int nlocal)
{
  return 2;
}


/* ----------------------------------------------------------------------
   Find minumum values of 4 given values 
------------------------------------------------------------------------- */
double FixLangevin::minvalue(double A, double B, double C, double D)
{
  double minval = 1.0e20;
  if (A < minval){
    minval = A;
  }
  if (B < minval){
    minval = B;
  }
  if (C < minval){
    minval = C;
  }
  if (D < minval){
    minval = D;
  }
  return minval; 
}

/* ----------------------------------------------------------------------
 * Overloaded 
   Find minumum values of 6 given values 
------------------------------------------------------------------------- */
double FixLangevin::minvalue(double A, double B, double C, double D, double E, double F)
{
  double minval = 1.0e20;
  if (A < minval){
    minval = A;
  }
  if (B < minval){
    minval = B;
  }
  if (C < minval){
    minval = C;
  }
  if (D < minval){
    minval = D;
  }
  if (E < minval){
    minval = E;
  }
  if (F < minval){
    minval = F;
  }
  return minval; 
}
