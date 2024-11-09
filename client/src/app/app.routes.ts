import { Routes } from '@angular/router';
import { HomePageComponent } from './pages/home-page/home-page.component';
import { DashboardComponent } from './pages/dashboard/dashboard.component';
import { AboutUsPageComponent } from './pages/about-us-page/about-us-page.component';
import { NotDonePageComponent } from './pages/not-done-page/not-done-page.component';
import { ScanTemplateComponent } from './pages/scan-template/scan-template.component';

export const routes: Routes = [
    { path: '', redirectTo: 'home', pathMatch: 'full' },
    { path: 'home', component: HomePageComponent },
    { path: 'dashboard', component: DashboardComponent },
    { path: 'about-us', component: AboutUsPageComponent },
    { path: 'model/:id', component: ScanTemplateComponent },
    { path: '**', redirectTo: 'not-found' },
    { path: 'not-found', component: NotDonePageComponent },
];
