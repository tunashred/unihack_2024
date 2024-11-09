import { Component } from '@angular/core';
import { FileUploaderComponent } from "../../components/file-uploader/file-uploader.component";
import { ActivatedRoute } from '@angular/router';
import { ScanTemplateService } from './scan-template.service';

@Component({
  selector: 'app-scan-template',
  standalone: true,
  imports: [FileUploaderComponent],
  templateUrl: './scan-template.component.html',
})
export class ScanTemplateComponent {
  constructor(private route: ActivatedRoute, private scanTemplateService: ScanTemplateService) {}
  public pageId: string | null = '';

  ngOnInit(): void {
    this.route.paramMap.subscribe(params => {
      this.pageId = params.get('id');
      console.warn(this.pageId)
    });

    this.getImage();
  }

  private getImage() {
    if (this.pageId) this.scanTemplateService.processImage('..\\models\\in\\kidney_tumor_0001.jpg').subscribe((res)=> console.warn(res));
  }
}
