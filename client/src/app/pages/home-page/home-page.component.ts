import { Component } from '@angular/core';
import { FileUploaderComponent } from "../../components/file-uploader/file-uploader.component";
import { HeroComponent } from "../../components/hero/hero.component";


@Component({
  selector: 'app-home-page',
  standalone: true,
  templateUrl: './home-page.component.html',
  styles: ``,
  imports: [FileUploaderComponent, HeroComponent]
})
export class HomePageComponent {

}
