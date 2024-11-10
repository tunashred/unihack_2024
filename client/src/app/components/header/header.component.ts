import { Component } from '@angular/core';
import { RouterLink, RouterModule, RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-header',
  standalone: true,
  imports:[RouterModule],
  templateUrl: './header.component.html',
  styles: ``
})
export class HeaderComponent {
  public scanConfig: { url: string, title: string} [] = [
    {url: 'home', title: 'Home page'},
    {url: 'dashboard', title: 'Dashboard'}, 
    {url: 'about-us', title: 'About us'}
   ];
}
