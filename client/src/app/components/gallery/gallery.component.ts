import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-gallery',
  standalone: true,
  imports: [],
  templateUrl: './gallery.component.html'
})
export class GalleryComponent implements OnInit {
  public items: {id: string, path: string}[]= [];

  
  ngOnInit() {
    this.items = []
  }
}
